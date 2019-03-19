#include "infer.h"
#include <opencv2/highgui/highgui.hpp>
#include <chrono>


std::vector<std::array<float, 4>> generateDefaultBox()
{
  std::vector<std::array<float, 4>> boxes;
  std::vector<int> feature_map_sizes = {32, 16, 8};
  for (auto& fmap : feature_map_sizes)
  {
    for (int h = 0; h < fmap; ++h)
    {
      for (int w = 0; w < fmap; ++w)
      {
        const float cx = (w + 0.5f)/(float)fmap;
        const float cy = (h + 0.5f)/(float)fmap;

        // Special multi aspect ratio code
        if (fmap == 32)
        {
            std::vector<std::vector<int>> densityMatrix = { { -3, -1, 1, 3 }, { -1, 1}, {0} };
            for (int j = 0; j < 3; ++j)
            {
              // size = current_aspect_ratio / fmap
              const float sar = (float)(1 << j) / fmap;

              for (auto dx : densityMatrix[j])
              {
                for (auto dy : densityMatrix[j])
                {
                  boxes.push_back({cx + (dx/8.0f) * sar, cy + (dy/8.0f) * sar, sar, sar});
                }
              }
            }
        }
        else
        {
          boxes.push_back({cx, cy, 4.0f / fmap, 4.0f / fmap});
        }
      }
    }
  }

  return boxes;
}

std::tuple<std::string, std::string> infer(
  cv::Mat image,
  int image_height, int image_width,
  std::shared_ptr<torch::jit::script::Module> model) {

  if (image.empty()) {
    std::cout << "WARNING: Cannot read image!" << std::endl;
  }

  std::string pred = "";
  std::string prob = "0.0";

  // Predict if image is not empty
  if (!image.empty()) {

    // Preprocess image
    cv::Mat imagergb = preprocess(image, image_width, image_height);

    std::vector<torch::Tensor> probs;

    // Forward
    probs = forward(imagergb, model);
    auto start = std::chrono::system_clock::now();
    for (int i =0; i < 100; ++i)
      probs = forward(imagergb, model);
    std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count() << "ms" << std::endl;

    int mult = std::max(imagergb.cols, imagergb.rows);

    // Boxes
    auto boxes = as_float(probs.at(0));


    // Grab confs
    auto [max_confs, labels] =  torch::max(probs.at(1), 1);
    auto maxIDs = as_long(torch::nonzero(labels).squeeze());
    
    auto conf = as_float(max_confs);

    auto probe = std::max_element(conf.begin(), conf.end());
    float prob_float = *probe;
    prob = std::to_string(prob_float);

    const float vxy = 0.1f;
    const float vwh = 0.2f;
    const float imwidth = imagergb.cols;
    const float imheight = imagergb.rows;

    auto defaultBox = generateDefaultBox();

    // Draw

    typedef struct BoxCoord {
      float x, y, w, h;
      float conf;
      std::tuple<float, float, float, float> scaled()
      {
        return { x * conf, y * conf, w * conf, h * conf };
      }
      BoxCoord average(BoxCoord other)
      {
          auto [bx, by, bw, bh] = scaled();
          auto [ox, oy, ow, oh] = other.scaled();
          auto sum = conf + other.conf;
          return BoxCoord {
            (bx + ox) / sum, (by + oy) / sum, (bw + ow) / sum, (bh + oh) / sum, sum / 2.0f
          };
      }
    } BoxCoord;

    std::vector<BoxCoord> coords;

    for (auto &x : maxIDs)
    {
      if (conf.at(x) < 0.8f)
        continue;

      std::vector<float> p = { boxes.begin() + x * 4, boxes.begin() + (x + 1) * 4 };
      const std::array<float, 4>& d = defaultBox.at(x);

      // Decode: How to accelerate this?

      // cX, cY
      for (int c = 0; c < 2; ++c)
      {
        p[c] = (p[c] * vxy * d[2+c] + d[c]);
      }
      // W, H
      for (int c = 2; c < 4; ++c)
      {
        p[c] = std::exp(p[c] * vwh) * d[c];
        p[c-2] -= p[c] / 2.0f;
      }

      BoxCoord tmp = {p[0] * (1024.0f / imwidth), p[1] * (1024.0f / imheight), p[2] * (1024.0f / imwidth), p[3] * (1024.0f / imheight), conf.at(x)};
      // NMS
      bool add = true;
      for (auto &b : coords)
      {
        // Does overlap?
        if (abs(b.x - tmp.x) < b.w &&
            abs(b.y - tmp.y) < b.h)
        {
          b = b.average(tmp);
          add = false;
          break;
        }
      }

      // Do we need to add item?
      if (add)
      {
        coords.push_back(tmp);
      }

    }
    for (const auto& b : coords)
    {
      cv::rectangle(image, cv::Rect{(int)(image.cols * b.x), (int)(image.rows * b.y), (int)(image.cols * b.w), (int)(image.rows * b.h)}, cv::Scalar{0,255,0}, 1);
    }
    cv::imshow("Frame", image);
    cv::waitKey(0);
  }

  return std::make_tuple(pred, prob);
}
