#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
int GetBestMatchingBBox(
    const int num_bboxes, const Dtype* gt, const Dtype* pred,
    const Dtype iou_threshold) {
  const Dtype xp = pred[0], yp = pred[1], wp = exp(pred[2]), hp = exp(pred[3]);
  const Dtype x1_p = xp - wp / 2, y1_p = yp - hp / 2;
  const Dtype x2_p = x1_p + wp, y2_p = y1_p + hp;
  Dtype max_iou = 0;
  Dtype max_iou_id = 0;
  Dtype max_area = 0;
  Dtype max_area_id = 0;
  for (int i = 0; i < num_bboxes; ++i) {
    const Dtype xg = gt[i * 4];
    const Dtype yg = gt[i * 4 + 1];
    const Dtype wg = exp(gt[i * 4 + 2]);
    const Dtype hg = exp(gt[i * 4 + 3]);
    const Dtype x1_g = xg - wg / 2, y1_g = yg - hg / 2;
    const Dtype x2_g = x1_g + wg, y2_g = y1_g + hg;
    const Dtype area = (x2_g - x1_g) * (y2_g - y1_g);
    if (area < max_area) {
      max_area = area;
      max_area_id = i;
    }
    const Dtype x1 = std::max(x1_p, x1_g);
    const Dtype x2 = std::min(x2_p, x2_g);
    const Dtype y1 = std::max(y1_p, y1_g);
    const Dtype y2 = std::min(y2_p, y2_g);
    if (x1 >= x2 || y1 >= y2) continue;
    const Dtype inter = (y2 - y1) * (x2 - x1);
    const Dtype sum = area + (y2_p - y1_p) * (x2_p - x1_p);
    const Dtype iou = inter / (sum - inter);
    if (iou > max_iou) {
      max_iou = iou;
      max_iou_id = i;
    }
  }
  return max_iou >= iou_threshold ? max_iou_id : max_area_id;
}

template <typename Dtype>
void ObjLocLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);

  const int num = bottom[0]->num();
  vector<int> shape(2);
  shape[0] = num;
  shape[1] = 4;
  pred_.Reshape(shape);
  gt_.Reshape(shape);

  iou_threshold_ = this->layer_param_.obj_loc_loss_param().iou_threshold();

  const string& loss_type = this->layer_param_.obj_loc_loss_param().loss_type();
  if (loss_type == "Smooth L1") {
    LayerParameter smooth_l1_param(this->layer_param_);
    smooth_l1_param.set_type("SmoothL1Loss");
    reg_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(smooth_l1_param);
    reg_loss_bottom_vec_.clear();
    reg_loss_bottom_vec_.push_back(&pred_);
    reg_loss_bottom_vec_.push_back(&gt_);
    reg_loss_top_vec_.push_back(top[0]);
    reg_loss_layer_->SetUp(reg_loss_bottom_vec_, reg_loss_top_vec_);
  } else {
    LOG(FATAL) << "Invalid loss type of object localization loss layer: "
               << loss_type;
  }
}

template <typename Dtype>
void ObjLocLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->num(), bottom[2]->num());
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[2]->channels(), 1);
  CHECK_EQ(bottom[2]->width(), 4);
  CHECK_EQ(bottom[3]->num(), bottom[0]->num());
  CHECK_EQ(bottom[3]->channels(), 1);

  const int num = bottom[0]->num();
  vector<int> shape(2);
  shape[0] = num;
  shape[1] = 4;
  pred_.Reshape(shape);
  gt_.Reshape(shape);
  reg_loss_layer_->Reshape(reg_loss_bottom_vec_, reg_loss_top_vec_);
}

template <typename Dtype>
void ObjLocLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // pcr = Per Class Regression, N x (Cx4)
  const Dtype* pcr = bottom[0]->cpu_data();
  // label, N x 1
  const Dtype* label = bottom[1]->cpu_data();
  // candidates, N x 1 x M x 4, where M is maximum number of bboxes
  const Dtype* candidates = bottom[2]->cpu_data();
  // candidates num, N x 1
  const Dtype* num_candidates = bottom[3]->cpu_data();
  // Get the bbox for the target category
  const int num = bottom[0]->num();
  const int dim = bottom[0]->count() / num;
  const int max_num_bboxes = bottom[2]->height();
  Dtype* pred_data = pred_.mutable_cpu_data();
  int offset = 0;
  for (int i = 0; i < num; ++i) {
    const int label_value = static_cast<int>(label[i]);
    for (int j = 0; j < 4; ++j) {
      pred_data[offset++] = pcr[i * dim + label_value * 4 + j];
    }
  }
  // Get the best matching ground truth candidate.
  // If the maximum IoU >= threshold, then use that ground truth bbox.
  // Otherwise use the bbox that has the largest area.
  Dtype* gt_data = gt_.mutable_cpu_data();
  offset = 0;
  for (int i = 0; i < num; ++i) {
    const int match_id = GetBestMatchingBBox(
        static_cast<int>(num_candidates[i]),
        candidates + i * max_num_bboxes * 4,
        pred_data + i * 4, iou_threshold_);
    for (int j = 0; j < 4; ++j) {
      gt_data[offset++] = candidates[i * max_num_bboxes * 4 + match_id * 4 + j];
    }
  }
  // // Debug outputs
  // if (Caffe::mpi_rank() == 0) {
  //   for (int i = 0; i < num; ++i) {
  //     printf("#%d\n", i);
  //     printf("Pred:\n");
  //     printf("%.5lf %.5lf %.5lf %.5lf\n", pred_data[i * 4], pred_data[i * 4 + 1], pred_data[i * 4 + 2], pred_data[i * 4 + 3]);
  //     printf("Cand:\n");
  //     const int num = static_cast<int>(num_candidates[i]);
  //     printf("%.5lf\n", num_candidates[i]);
  //     for (int j = 0; j < num; ++j) {
  //       printf("%.5lf %.5lf %.5lf %.5lf\n",
  //           candidates[i * max_num_bboxes * 4 + j * 4],
  //           candidates[i * max_num_bboxes * 4 + j * 4 + 1],
  //           candidates[i * max_num_bboxes * 4 + j * 4 + 2],
  //           candidates[i * max_num_bboxes * 4 + j * 4 + 3]);
  //     }
  //     printf("GT:\n");
  //     printf("%.5lf %.5lf %.5lf %.5lf\n", gt_data[i * 4], gt_data[i * 4 + 1], gt_data[i * 4 + 2], gt_data[i * 4 + 3]);
  //   }
  //   printf("\n");
  // }
  // The forward pass of the regression loss layer
  reg_loss_layer_->Forward(reg_loss_bottom_vec_, reg_loss_top_vec_);
}

template <typename Dtype>
void ObjLocLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[2]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to ground truth bboxes.";
  }
  if (propagate_down[0]) {
    const int num = bottom[0]->num();
    const int dim = bottom[0]->count() / num;
    Dtype* pcr_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* label = bottom[1]->cpu_data();
    const Dtype* pred_diff = pred_.mutable_cpu_diff();
    caffe_set<Dtype>(bottom[0]->count(), 0, pcr_diff);
    for (int i = 0; i < num; ++i) {
      const int label_value = static_cast<int>(label[i]);
      for (int j = 0; j < 4; ++j) {
        pcr_diff[i * dim + label_value * 4 + j] = pred_diff[i * 4 + j];
      }
    }
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(bottom[0]->count(), loss_weight / num, pcr_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(ObjLocLossLayer);
#endif

INSTANTIATE_CLASS(ObjLocLossLayer);
REGISTER_LAYER_CLASS(ObjLocLoss);

}  // namespace caffe
