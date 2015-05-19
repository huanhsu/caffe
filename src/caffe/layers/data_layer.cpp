#include <opencv2/core/core.hpp>

#include <stdint.h>

#include <string>
#include <vector>
#ifdef USE_MPI
#include <mpi.h>
#endif

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
DataLayer<Dtype>::~DataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void DataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Initialize DB
  db_.reset(db::GetDB(this->layer_param_.data_param().backend()));
  db_->Open(this->layer_param_.data_param().source(), db::READ);
  cursor_.reset(db_->NewCursor());

  // Check if we should randomly skip a few data points
  unsigned int skip = 0;
  if (this->layer_param_.data_param().rand_skip()) {
    skip = caffe_rng_rand() % this->layer_param_.data_param().rand_skip();
  }
#ifdef USE_MPI
  MPI_Bcast(&skip, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  skip += this->layer_param_.data_param().batch_size() * Caffe::mpi_rank();
#endif
  LOG(INFO) << "Skipping first " << skip << " data points.";
  while (skip-- > 0) {
    cursor_->Next();
    if (!cursor_->valid()) {
      cursor_->SeekToFirst();
    }
  }
  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  datum.ParseFromString(cursor_->value());

  bool force_color = this->layer_param_.data_param().force_encoded_color();
  if ((force_color && DecodeDatum(&datum, true)) ||
      DecodeDatumNative(&datum)) {
    LOG(INFO) << "Decoding Datum";
  }
  // image
  int crop_size = this->layer_param_.transform_param().crop_size();
  int crop_height = this->layer_param_.transform_param().crop_height() > 0 ?
      this->layer_param_.transform_param().crop_height() : crop_size;
  int crop_width = this->layer_param_.transform_param().crop_width() > 0 ?
      this->layer_param_.transform_param().crop_width() : crop_size;

  if (crop_height || crop_width) {
    top[0]->Reshape(this->layer_param_.data_param().batch_size(),
        datum.channels(), crop_height, crop_width);
    this->prefetch_data_.Reshape(this->layer_param_.data_param().batch_size(),
        datum.channels(), crop_height, crop_width);
    this->transformed_data_.Reshape(1, datum.channels(), crop_height, crop_width);
  } else {
    top[0]->Reshape(
        this->layer_param_.data_param().batch_size(), datum.channels(),
        datum.height(), datum.width());
    this->prefetch_data_.Reshape(this->layer_param_.data_param().batch_size(),
        datum.channels(), datum.height(), datum.width());
    this->transformed_data_.Reshape(1, datum.channels(),
      datum.height(), datum.width());
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    vector<int> label_shape(1, this->layer_param_.data_param().batch_size());
    top[1]->Reshape(label_shape);
    this->prefetch_label_.Reshape(label_shape);
  }
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void DataLayer<Dtype>::InternalThreadEntry() {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(this->prefetch_data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape on single input batches for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  const int crop_size = this->layer_param_.transform_param().crop_size();
  int crop_height = this->layer_param_.transform_param().crop_height() > 0 ?
        this->layer_param_.transform_param().crop_height() : crop_size;
  int crop_width = this->layer_param_.transform_param().crop_width() > 0 ?
        this->layer_param_.transform_param().crop_width() : crop_size;
  bool force_color = this->layer_param_.data_param().force_encoded_color();
  if (batch_size == 1 && crop_height == 0 && crop_width) {
    Datum datum;
    datum.ParseFromString(cursor_->value());
    if (datum.encoded()) {
      if (force_color) {
        DecodeDatum(&datum, true);
      } else {
        DecodeDatumNative(&datum);
      }
    }
    this->prefetch_data_.Reshape(1, datum.channels(),
        datum.height(), datum.width());
    this->transformed_data_.Reshape(1, datum.channels(),
        datum.height(), datum.width());
  }

  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  if (this->output_labels_) {
    top_label = this->prefetch_label_.mutable_cpu_data();
  }
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a blob
    Datum datum;
    datum.ParseFromString(cursor_->value());

    cv::Mat cv_img;
    if (datum.encoded()) {
      if (force_color) {
        cv_img = DecodeDatumToCVMat(datum, true);
      } else {
        cv_img = DecodeDatumToCVMatNative(datum);
      }
      if (cv_img.channels() != this->transformed_data_.channels()) {
        LOG(WARNING) << "Your dataset contains encoded images with mixed "
        << "channel sizes. Consider adding a 'force_color' flag to the "
        << "model definition, or rebuild your dataset using "
        << "convert_imageset.";
      }
    }
    read_time += timer.MicroSeconds();
    timer.Start();

    // Apply data transformations (mirror, scale, crop...)
    int offset = this->prefetch_data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    if (datum.encoded()) {
      this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    } else {
      this->data_transformer_->Transform(datum, &(this->transformed_data_));
    }
    if (this->output_labels_) {
      top_label[item_id] = datum.label();
    }
    trans_time += timer.MicroSeconds();
    // go to the next iter
    cursor_->Next();
    if (!cursor_->valid()) {
      DLOG(INFO) << "Restarting data prefetching from start.";
      cursor_->SeekToFirst();
    }
  }
#ifdef USE_MPI
  for (int i = 0; i < batch_size * (Caffe::mpi_size() - 1); ++i) {
    cursor_->Next();
    if (!cursor_->valid()) {
      cursor_->SeekToFirst();
    }
  }
#endif
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(DataLayer);
REGISTER_LAYER_CLASS(Data);

}  // namespace caffe
