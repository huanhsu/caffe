#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
bool HasMatchingBBox(
    const int num_bboxes, const Dtype* gt, const Dtype* pred,
    const Dtype iou_threshold) {
  const Dtype xp = pred[0], yp = pred[1], wp = exp(pred[2]), hp = exp(pred[3]);
  const Dtype x1_p = xp - wp / 2, y1_p = yp - hp / 2;
  const Dtype x2_p = x1_p + wp, y2_p = y1_p + hp;
  for (int i = 0; i < num_bboxes; ++i) {
    const Dtype xg = gt[i * 4];
    const Dtype yg = gt[i * 4 + 1];
    const Dtype wg = exp(gt[i * 4 + 2]);
    const Dtype hg = exp(gt[i * 4 + 3]);
    const Dtype x1_g = xg - wg / 2, y1_g = yg - hg / 2;
    const Dtype x2_g = x1_g + wg, y2_g = y1_g + hg;
    const Dtype area = (x2_g - x1_g) * (y2_g - y1_g);
    const Dtype x1 = std::max(x1_p, x1_g);
    const Dtype x2 = std::min(x2_p, x2_g);
    const Dtype y1 = std::max(y1_p, y1_g);
    const Dtype y2 = std::min(y2_p, y2_g);
    if (x1 >= x2 || y1 >= y2) continue;
    const Dtype inter = (y2 - y1) * (x2 - x1);
    const Dtype sum = area + (y2_p - y1_p) * (x2_p - x1_p);
    const Dtype iou = inter / (sum - inter);
    if (iou >= iou_threshold) return true;
  }
  return false;
}

template <typename Dtype>
void ObjLocAccuracyLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top_k_ = this->layer_param_.accuracy_param().top_k();
  iou_threshold_ = this->layer_param_.accuracy_param().iou_threshold();
}

template <typename Dtype>
void ObjLocAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->num(), bottom[2]->num());
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[2]->channels(), 1);
  CHECK_EQ(bottom[2]->width(), 4);
  CHECK_EQ(bottom[3]->num(), bottom[0]->num());
  CHECK_EQ(bottom[3]->channels(), 1);
  if (bottom.size() == 5) {
    CHECK_EQ(bottom[4]->num(), bottom[0]->num());
    CHECK_EQ(bottom[4]->channels(), 1);
    CHECK_EQ(bottom[4]->height(), top_k_);
    CHECK_EQ(bottom[4]->width(), 1);
  }
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void ObjLocAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // pcr = Per Class Regression, N x (Cx4)
  const Dtype* pcr = bottom[0]->cpu_data();
  // label, N x 1
  const Dtype* label = bottom[1]->cpu_data();
  // candidates, N x 1 x M x 4, where M is maximum number of bboxes
  const Dtype* candidates = bottom[2]->cpu_data();
  // candidates num, N x 1
  const Dtype* num_candidates = bottom[3]->cpu_data();
  const int num = bottom[0]->num();
  const int dim = bottom[0]->count() / num;
  const int max_num_bboxes = bottom[2]->height();
  Dtype accuracy = 0;
  if (bottom.size() == 4) {
    // Get the bbox for the target category
    for (int i = 0; i < num; ++i) {
      const int label_value = static_cast<int>(label[i]);
      accuracy += HasMatchingBBox(
          static_cast<int>(num_candidates[i]),
          candidates + i * max_num_bboxes * 4,
          pcr + i * dim + label_value * 4, iou_threshold_);
    }
  } else {
    // Get the bbox for each predicted category
    // Top-k classification prediction, N x 1 x K x 1
    const Dtype* top_k_pred = bottom[4]->cpu_data();
    for (int i = 0; i < num; ++i) {
      const int label_value = static_cast<int>(label[i]);
      for (int k = 0; k < top_k_; ++k) {
        const int pred_label = static_cast<int>(top_k_pred[i * top_k_ + k]);
        if (pred_label != label_value) continue;
        accuracy += HasMatchingBBox(
            static_cast<int>(num_candidates[i]),
            candidates + i * max_num_bboxes * 4,
            pcr + i * dim + pred_label * 4, iou_threshold_);
      }
    }
  }
  top[0]->mutable_cpu_data()[0] = accuracy / num;
}

INSTANTIATE_CLASS(ObjLocAccuracyLayer);
REGISTER_LAYER_CLASS(ObjLocAccuracy);

}  // namespace caffe
