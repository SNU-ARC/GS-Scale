#ifndef _CPU_ADAM_H
#define _CPU_ADAM_H

#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <cmath>

at::Tensor quat_to_rotmat(at::Tensor& quats);

std::tuple<at::Tensor, at::Tensor> persp_proj(at::Tensor& means, at::Tensor& covars, at::Tensor& Ks, int width, int height);

std::tuple<torch::Tensor, torch::Tensor> world_to_cam(torch::Tensor& means, torch::Tensor& covars, torch::Tensor& viewmats);

at::Tensor calculate_update_ids(at::Tensor valid_ids, at::Tensor counter, int total_num);

void update_counter(at::Tensor counter, at::Tensor update_ids);

void adam_deferred_update(at::Tensor weight, at::Tensor grad, at::Tensor exp_avg, at::Tensor exp_avg_sq, at::Tensor valid_ids, at::Tensor counter,
    int step, float lr, float beta1, float beta2, float eps);

void adam_for_next_with_counter(at::Tensor weight, at::Tensor grad, at::Tensor exp_avg, at::Tensor exp_avg_sq, at::Tensor valid_ids, at::Tensor weight_new, at::Tensor counter,
    int step, float lr, float beta1, float beta2, float eps);

void sparse_adam(at::Tensor weight, at::Tensor grad, at::Tensor exp_avg, at::Tensor exp_avg_sq, at::Tensor valid_ids,
    int step, float lr, float beta1, float beta2, float eps);

void adam_for_next(at::Tensor weight, at::Tensor grad, at::Tensor exp_avg, at::Tensor exp_avg_sq, at::Tensor valid_ids, at::Tensor weight_new,
    int step, float lr, float beta1, float beta2, float eps);

void index_copy(at::Tensor src, at::Tensor indices, at::Tensor dest);

#endif
