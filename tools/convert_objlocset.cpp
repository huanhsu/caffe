// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_objlocset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7 2 0 0 500 500 2 4 100 70
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;
using std::vector;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_int32(resize_short_side, 0,
    "Resize short side to this value and keep aspect ratio");
DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
    "When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
    "Optional: What type should we encode the image as ('png','jpg',...).");


template <typename T>
struct BBox {
  T x, y, width, height;
};

template <typename T>
struct LineItem {
  string filename;
  int label;
  vector<BBox<T> > bboxes;
};

template <typename T>
bool SetBBoxesToDatum(const vector<BBox<T> >& bboxes,
                      const int height, const int width, Datum* datum) {
  // Convert the bboxes into (x_c, y_c, w, h), where x_c, y_c are relative
  // center coordinates, and w, h are width and height in log space.
  // Store the converted bboxes into a float vector, then put into
  // datum.float_data.
  if (bboxes.size() <= 0) return false;
  vector<float> bbox_data(bboxes.size() * 4, 0);
  for (int i = 0; i < bboxes.size(); ++i) {
    const BBox<T>& bbox = bboxes[i];
    float xc = (float)(bbox.x + bbox.width / 2) / width;
    float yc = (float)(bbox.y + bbox.height / 2) / height;
    float w = log((float)(bbox.width) / width);
    float h = log((float)(bbox.height) / height);
    bbox_data[i * 4] = xc;
    bbox_data[i * 4 + 1] = yc;
    bbox_data[i * 4 + 2] = w;
    bbox_data[i * 4 + 3] = h;
  }
  datum->clear_float_data();
  for (int d = 0; d < bbox_data.size(); ++d) {
    datum->add_float_data(bbox_data[d]);
  }
  return true;
}

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_objlocset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n"
        "The ImageNet dataset for the training demo is at\n"
        "    http://www.image-net.org/download-images\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_objlocset");
    return 1;
  }

  const bool is_color = !FLAGS_gray;
  const bool check_size = FLAGS_check_size;
  const bool encoded = FLAGS_encoded;
  const string encode_type = FLAGS_encode_type;

  std::ifstream infile(argv[2]);
  LineItem<int> item;
  int count = 0;
  vector<LineItem<int> > lines;
  while (infile >> item.filename >> item.label >> count) {
    item.bboxes.resize(count);
    for (int i = 0; i < count; ++i) {
      infile >> item.bboxes[i].x >> item.bboxes[i].y
             >> item.bboxes[i].width >> item.bboxes[i].height;
    }
    lines.push_back(item);
  }
  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";

  if (encode_type.size() && !encoded)
    LOG(INFO) << "encode_type specified, assuming encoded=true.";

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);
  int resize_short_side = std::max<int>(0, FLAGS_resize_short_side);

  // Create new DB
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(argv[3], db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

  // Storing to db
  std::string root_folder(argv[1]);
  Datum datum;
  count = 0;
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];
  int data_size = 0;
  bool data_size_initialized = false;

  cv::Mat cv_img;
  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    const LineItem<int>& item = lines[line_id];
    bool status;
    std::string enc = encode_type;
    if (encoded && !enc.size()) {
      // Guess the encoding type from the file name
      string fn = item.filename;
      size_t p = fn.rfind('.');
      if ( p == fn.npos )
        LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
      enc = fn.substr(p);
      std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
    }
    if (resize_short_side > 0) {
      status = ReadImageToDatumResizeShortSide(
          root_folder + item.filename,
          item.label, resize_short_side, is_color,
          enc, &datum);
    } else {
      status = ReadImageToDatum(root_folder + item.filename,
          item.label, resize_height, resize_width, is_color,
          enc, &datum);
    }
    // Get original image size and compute relative bbox
    cv_img = cv::imread(root_folder + item.filename, CV_LOAD_IMAGE_COLOR);
    status = SetBBoxesToDatum(item.bboxes, cv_img.rows, cv_img.cols, &datum);
    if (status == false) continue;
    if (check_size) {
      if (!data_size_initialized) {
        data_size = datum.channels() * datum.height() * datum.width();
        data_size_initialized = true;
      } else {
        const std::string& data = datum.data();
        CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
            << data.size();
      }
    }
    // sequential
    int length = snprintf(key_cstr, kMaxKeyLength, "%08d_%s", line_id,
        item.filename.c_str());

    // Put in db
    string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(string(key_cstr, length), out);

    if (++count % 1000 == 0) {
      // Commit db
      txn->Commit();
      txn.reset(db->NewTransaction());
      LOG(ERROR) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(ERROR) << "Processed " << count << " files.";
  }
  return 0;
}
