#include <opencv2/core/core.hpp>

#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/mpi_templates.hpp"

namespace caffe {

template <typename Dtype>
ObjLocDataLayer<Dtype>::~ObjLocDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void ObjLocDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
#ifdef USE_MPI
  if (Caffe::mpi_size() > 1) {
    if (this->layer_param_.data_param().batch_size() % Caffe::mpi_size() != 0) {
      LOG(FATAL) << "Batch size (" << this->layer_param_.data_param().batch_size()
                 << ") should be divisible by the number of MPI processes ("
                 << Caffe::mpi_size() << ")";
    }
    this->layer_param_.mutable_data_param()->set_batch_size(
        this->layer_param_.data_param().batch_size() / Caffe::mpi_size());
  }
#endif
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
  if (Caffe::mpi_size() > 1) {
    MPIBcast<unsigned int>(1, &skip);
    skip += this->layer_param_.data_param().batch_size() * Caffe::mpi_rank();
  }
#endif
  LOG(INFO) << "Skipping first " << skip << " data points.";
  while (skip-- > 0) {
    cursor_->Next();
    if (!cursor_->valid()) {
      cursor_->SeekToFirst();
    }
  }
  // Read a data point, to initialize the prefetch and top blobs.
  Datum datum;
  datum.ParseFromString(cursor_->value());
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = this->layer_param_.data_param().batch_size();
  this->prefetch_data_.Reshape(top_shape);
  top[0]->ReshapeLike(this->prefetch_data_);

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
  this->output_labels_ = (top.size() >= 2);
  if (this->output_labels_) {
    vector<int> label_shape(1, this->layer_param_.data_param().batch_size());
    top[1]->Reshape(label_shape);
    this->prefetch_label_.Reshape(label_shape);
  }
  // bboxes
  this->output_bboxes_ = (top.size() >= 3);
  if (this->output_bboxes_) {
    this->max_num_bboxes_ = this->layer_param_.data_param().max_num_bboxes();
    if (this->max_num_bboxes_ <= 0) {
      LOG(FATAL) << "Must specify the maximum number of bboxes";
    }
    vector<int> bboxes_shape(4);
    bboxes_shape[0] = this->layer_param_.data_param().batch_size();
    bboxes_shape[1] = 1;
    bboxes_shape[2] = this->max_num_bboxes_;
    bboxes_shape[3] = 4;
    top[2]->Reshape(bboxes_shape);
    this->prefetch_bboxes_.Reshape(bboxes_shape);
    bboxes_shape[0] = 1;
    this->transformed_bboxes_.Reshape(bboxes_shape);
  }
  this->output_num_bboxes_ = (top.size() >= 4);
  if (this->output_num_bboxes_) {
    vector<int> shape(1, this->layer_param_.data_param().batch_size());
    top[3]->Reshape(shape);
    this->prefetch_num_bboxes_.Reshape(shape);
  }
  // Initialize the shuffle pool index
  const int shuffle_pool_size =
      this->layer_param_.data_param().shuffle_pool_size();
  if (shuffle_pool_size > 1) {
    shuffle_pool_index_.resize(shuffle_pool_size *
        this->layer_param_.data_param().batch_size());
    for (int i = 0; i < shuffle_pool_index_.size(); ++i) {
      shuffle_pool_index_[i] = i;
    }
  }
}

template <typename Dtype>
void ObjLocDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  this->JoinPrefetchThread();
  DLOG(INFO) << "Thread joined";
  // Reshape to loaded data.
  top[0]->ReshapeLike(this->prefetch_data_);
  // Copy the data
  caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
             top[0]->mutable_cpu_data());
  DLOG(INFO) << "Prefetch data copied";
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(this->prefetch_label_);
    // Copy the labels.
    caffe_copy(this->prefetch_label_.count(), this->prefetch_label_.cpu_data(),
               top[1]->mutable_cpu_data());
  }
  DLOG(INFO) << "Prefetch label copied";
  if (this->output_bboxes_) {
    // Reshape to loaded bboxes.
    top[2]->ReshapeLike(this->prefetch_bboxes_);
    // Copy the bboxes.
    caffe_copy(this->prefetch_bboxes_.count(),
               this->prefetch_bboxes_.cpu_data(),
               top[2]->mutable_cpu_data());
  }
  if (this->output_num_bboxes_) {
    top[3]->ReshapeLike(this->prefetch_num_bboxes_);
    caffe_copy(this->prefetch_num_bboxes_.count(),
               this->prefetch_num_bboxes_.cpu_data(),
               top[3]->mutable_cpu_data());
  }
  DLOG(INFO) << "Prefetch bboxes copied";
  // Start a new prefetch thread
  DLOG(INFO) << "CreatePrefetchThread";
  this->CreatePrefetchThread();
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void ObjLocDataLayer<Dtype>::InternalThreadEntry() {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(this->prefetch_data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  const int shuffle_pool_size =
      this->layer_param_.data_param().shuffle_pool_size();
  Datum datum;
  datum.ParseFromString(cursor_->value());
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  this->prefetch_data_.Reshape(top_shape);

  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  Dtype* top_bboxes = NULL;
  Dtype* top_num_bboxes = NULL;

  if (this->output_labels_) {
    top_label = this->prefetch_label_.mutable_cpu_data();
  }
  if (this->output_bboxes_) {
    top_bboxes = this->prefetch_bboxes_.mutable_cpu_data();
  }
  if (this->output_num_bboxes_) {
    top_num_bboxes = this->prefetch_num_bboxes_.mutable_cpu_data();
  }
  timer.Start();
  const bool is_shuffle_pool_full = (shuffle_pool_size > 1
      && shuffle_pool_.size() >= shuffle_pool_size * batch_size);
  if (is_shuffle_pool_full) {
    shuffle(shuffle_pool_index_.begin(), shuffle_pool_index_.end());
  }
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // Get a datum to be transformed either from DB or the shuffle pool.
    Datum datum;
    if (is_shuffle_pool_full) {
      int pool_index = shuffle_pool_index_[item_id];
      datum = shuffle_pool_[pool_index];
      shuffle_pool_[pool_index].ParseFromString(cursor_->value());
    } else {
      datum.ParseFromString(cursor_->value());
      if (shuffle_pool_size > 1) { // Ths shuffle pool is not full.
        shuffle_pool_.push_back(datum);
      }
    }
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply data transformations (mirror, scale, crop...)
    int offset = this->prefetch_data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    if (!this->output_bboxes_) {
      this->data_transformer_->Transform(datum, &(this->transformed_data_));
    } else {
      offset = this->prefetch_bboxes_.offset(item_id);
      this->transformed_bboxes_.set_cpu_data(top_bboxes + offset);
      this->data_transformer_->TransformObjLoc(datum,
          &(this->transformed_data_), &(this->transformed_bboxes_));
    }
    // Copy label.
    if (this->output_labels_) {
      top_label[item_id] = datum.label();
    }
    if (this->output_num_bboxes_) {
      top_num_bboxes[item_id] = datum.float_data_size() / 4;
    }
    trans_time += timer.MicroSeconds();
    timer.Start();
    // go to the next item.
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
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(ObjLocDataLayer, Forward);
#endif

INSTANTIATE_CLASS(ObjLocDataLayer);
REGISTER_LAYER_CLASS(ObjLocData);

}  // namespace caffe
