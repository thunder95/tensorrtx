//
// Created by hl on 22-3-1.
//

#ifndef TRTPOSE_POST_PROCESS_H
#define TRTPOSE_POST_PROCESS_H

#pragma once
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#include "cmath"
#include "cover_table.h"
#include "pair_graph.h"
#include <cstring>
#include <queue>


inline int reflect(int idx, int min, int max) {
    if (idx < min) {
        return -idx;
    } else if (idx >= max) {
        return max - (idx - max) - 2;
    } else {
        return idx;
    }
}

//查找峰值, 后期使用cuda实现
/* 1个批次, 假定最多30个人体姿态
 * count 18个整数, 统计peak的数量
 * peak 18*30*2, 暂存peak的值
 */
void find_peaks(float *input, int *counts, float *peaks, float *threshs, const float threshold = 0.1,
        const int H = 56, const int W = 56, const int window_size = 5, const int C = 18,  const int M = 30) {

    int win = window_size / 2;

    //hw范围内寻值
    for (int c = 0; c < C; c++) {
        int offset = c * H * W;
        int p_offset = c * M * 2;
        int count = 0;

        for (int i = 0; i < H && count < M; i++) {
            for (int j = 0; j < W && count < M; j++) {
                float val = input[offset + i * W + j];

                // skip if below threshold
                if (val < threshold)
                    continue;
                threshs[count] = val;

                // compute window bounds
                int ii_min = MAX(i - win, 0);
                int jj_min = MAX(j - win, 0);
                int ii_max = MIN(i + win + 1, H);
                int jj_max = MIN(j + win + 1, W);

                // search for larger value in window ===>判断此点是否是周边的最大值
                bool is_peak = true;
                for (int ii = ii_min; ii < ii_max; ii++) {
                    for (int jj = jj_min; jj < jj_max; jj++) {
                        if (input[offset + ii * W + jj] > val) {
                            is_peak = false;
                        }
                    }
                }

                // add peak ====> 添加峰值点
                if (is_peak) {
//                    peaks[p_offset + count * 2] = i;
//                    peaks[p_offset + count * 2 + 1] = j;


                    //计算区域权重
                    float weight_sum = 0.0;
                    float *peak_1 = &peaks[p_offset + count * 2];
                    float *peak_2 = &peaks[p_offset + count * 2 + 1];
                    for (int ii = i - win; ii < i + win + 1; ii++) {
                        int ii_idx = reflect(ii, 0, H);
                        for (int jj = j - win; jj < j + win + 1; jj++) {
                            int jj_idx = reflect(jj, 0, W);
                            float weight = input[offset + ii_idx * W + jj_idx];
                            //std::cout<<"weight: "<<ii_idx << "->"<<jj_idx<<"===>"<<weight<<std::endl;
                            *peak_1 += weight * ii;
                            *peak_2 += weight * jj;
                            weight_sum += weight;
                        }
                    }
                    //std::cout<<"peak: "<<i<<"->"<<j<<"===>"<<weight_sum<<std::endl;
                    *peak_1 /= weight_sum;
                    *peak_2 /= weight_sum;
                    *peak_1 += 0.5; // center pixel
                    *peak_2 += 0.5; // center pixel
                    *peak_1 /= H;   // normalize coordinates
                    *peak_2 /= W;   // normalize coordinates

                    count++;
                }
            }

        }
//        std::cout <<"c: "<< c << "count: " << count << std::endl;
        counts[c] = count;
    }
}

//计算分值
void paf_score_graph_out_hw(float *score_graph, // MxM
                            const float *paf_i, // HxW
                            const float *paf_j, // HxW
                            const int counts_a, const int counts_b,
                            const float *peaks_a, // Mx2
                            const float *peaks_b, // Mx2
                            const int H, const int W, const int M,
                            const int num_integral_samples) {
    for (int a = 0; a < counts_a; a++) {
        // compute point A
        float pa_i = peaks_a[a * 2] * H;
        float pa_j = peaks_a[a * 2 + 1] * W;

        for (int b = 0; b < counts_b; b++) {
            // compute point B
            float pb_i = peaks_b[b * 2] * H;
            float pb_j = peaks_b[b * 2 + 1] * W;

            // compute vector A->B
            float pab_i = pb_i - pa_i;
            float pab_j = pb_j - pa_j;

            // compute normalized vector A->B
            float pab_norm = sqrtf(pab_i * pab_i + pab_j * pab_j) + 1e-5;
            float uab_i = pab_i / pab_norm;
            float uab_j = pab_j / pab_norm;

            float integral = 0.;
            float increment = 1.f / num_integral_samples;

            for (int t = 0; t < num_integral_samples; t++) {
                // compute integral point T
                float progress = (float) t / ((float) num_integral_samples - 1);
                float pt_i = pa_i + progress * pab_i;
                float pt_j = pa_j + progress * pab_j;

                // convert to int
                // note: we do not need to subtract 0.5 when indexing, because
                // round(x - 0.5) = int(x)
                int pt_i_int = (int) pt_i;
                int pt_j_int = (int) pt_j;

                // skip point if out of bounds (will weaken integral)
                if (pt_i_int < 0)
                    continue;
                if (pt_i_int >= H)
                    continue;
                if (pt_j_int < 0)
                    continue;
                if (pt_j_int >= W)
                    continue;

                // get vector at integral point from PAF
                float pt_paf_i = paf_i[pt_i_int * W + pt_j_int];
                float pt_paf_j = paf_j[pt_i_int * W + pt_j_int];

                // compute dot product of normalized A->B with PAF vector at integral
                // point
                float dot = pt_paf_i * uab_i + pt_paf_j * uab_j;
                integral += dot;
            }

            integral /= num_integral_samples;
            score_graph[a * M + b] = integral;
        }
    }
}

void paf_score_graph(const float *paf, const float *peaks, float *score_graph, const int *topology, const int *counts,
                     const int H = 56, const int W = 56, const int K = 21, const int M = 30,
                     const int num_integral_samples = 7) {
    for (int k = 0; k < K; k++) {
        float *score_graph_k = &score_graph[k * M * M];
        const int *tk = &topology[k * 4];
        const int paf_i_idx = tk[0];
        const int paf_j_idx = tk[1];
        const int cmap_a_idx = tk[2];
        const int cmap_b_idx = tk[3];
        const float *paf_i = &paf[paf_i_idx * H * W];
        const float *paf_j = &paf[paf_j_idx * H * W];

        const int counts_a = counts[cmap_a_idx];
        const int counts_b = counts[cmap_b_idx];
        const float *peaks_a = &peaks[cmap_a_idx * M * 2];
        const float *peaks_b = &peaks[cmap_b_idx * M * 2];

        paf_score_graph_out_hw(score_graph_k, paf_i, paf_j, counts_a, counts_b,
                               peaks_a, peaks_b, H, W, M, num_integral_samples);
    }


}


//-------------------munres计算连接线---------------//
void subMinRow(float *cost_graph, const int M, const int nrows,
               const int ncols) {
    for (int i = 0; i < nrows; i++) {
        // find min
        float min = cost_graph[i * M];
        for (int j = 0; j < ncols; j++) {
            float val = cost_graph[i * M + j];
            if (val < min) {
                min = val;
            }
        }

        // subtract min
        for (int j = 0; j < ncols; j++) {
            cost_graph[i * M + j] -= min;
        }
    }
}

void subMinCol(float *cost_graph, const int M, const int nrows,
               const int ncols) {
    for (int j = 0; j < ncols; j++) {
        // find min
        float min = cost_graph[j];
        for (int i = 0; i < nrows; i++) {
            float val = cost_graph[i * M + j];
            if (val < min) {
                min = val;
            }
        }

        // subtract min
        for (int i = 0; i < nrows; i++) {
            cost_graph[i * M + j] -= min;
        }
    }
}

void munkresStep1(const float *cost_graph, const int M, PairGraph &star_graph,
                  const int nrows, const int ncols) {
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            if (!star_graph.isRowSet(i) && !star_graph.isColSet(j) &&
                (cost_graph[i * M + j] == 0)) {
                star_graph.set(i, j);
            }
        }
    }
}

// returns 1 if we should exit
bool munkresStep2(const PairGraph &star_graph, TrtPoseCoverTable &cover_table) {
    int k =
            star_graph.nrows < star_graph.ncols ? star_graph.nrows : star_graph.ncols;
    int count = 0;
    for (int j = 0; j < star_graph.ncols; j++) {
        if (star_graph.isColSet(j)) {
            cover_table.coverCol(j);
            count++;
        }
    }
    return count >= k;
}

bool munkresStep3(const float *cost_graph, const int M,
                  const PairGraph &star_graph, PairGraph &prime_graph,
                  TrtPoseCoverTable &cover_table, std::pair<int, int> &p,
                  const int nrows, const int ncols) {
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            if (cost_graph[i * M + j] == 0 && !cover_table.isCovered(i, j)) {
                prime_graph.set(i, j);
                if (star_graph.isRowSet(i)) {
                    cover_table.coverRow(i);
                    cover_table.uncoverCol(star_graph.colForRow(i));
                } else {
                    p.first = i;
                    p.second = j;
                    return 1;
                }
            }
        }
    }
    return 0;
};

void munkresStep4(PairGraph &star_graph, PairGraph &prime_graph,
                  TrtPoseCoverTable &cover_table, std::pair<int, int> p) {
    // repeat until no star found in prime's column
    while (star_graph.isColSet(p.second)) {
        // find and reset star in prime's column
        std::pair<int, int> s = {star_graph.rowForCol(p.second), p.second};
        star_graph.reset(s.first, s.second);

        // set this prime to a star
        star_graph.set(p.first, p.second);

        // repeat for prime in cleared star's row
        p = {s.first, prime_graph.colForRow(s.first)};
    }
    star_graph.set(p.first, p.second);
    cover_table.clear();
    prime_graph.clear();
}

void munkresStep5(float *cost_graph, const int M, const TrtPoseCoverTable &cover_table,
                  const int nrows, const int ncols) {
    bool valid = false;
    float min;
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            if (!cover_table.isCovered(i, j)) {
                if (!valid) {
                    min = cost_graph[i * M + j];
                    valid = true;
                } else if (cost_graph[i * M + j] < min) {
                    min = cost_graph[i * M + j];
                }
            }
        }
    }

    for (int i = 0; i < nrows; i++) {
        if (cover_table.isRowCovered(i)) {
            for (int j = 0; j < ncols; j++) {
                cost_graph[i * M + j] += min;
            }
            //       cost_graph.addToRow(i, min);
        }
    }
    for (int j = 0; j < ncols; j++) {
        if (!cover_table.isColCovered(j)) {
            for (int i = 0; i < nrows; i++) {
                cost_graph[i * M + j] -= min;
            }
            //       cost_graph.addToCol(j, -min);
        }
    }
}

void _munkres(float *cost_graph, const int M, PairGraph &star_graph,
              const int nrows, const int ncols) {
    PairGraph prime_graph(nrows, ncols);
    TrtPoseCoverTable cover_table(nrows, ncols);
    prime_graph.clear();
    cover_table.clear();
    star_graph.clear();

    int step = 0;
    if (ncols >= nrows) {
        subMinRow(cost_graph, M, nrows, ncols);
    }
    if (ncols > nrows) {
        step = 1;
    }

    std::pair<int, int> p;
    bool done = false;
    while (!done) {
        switch (step) {
            case 0:
                subMinCol(cost_graph, M, nrows, ncols);
            case 1:
                munkresStep1(cost_graph, M, star_graph, nrows, ncols);
            case 2:
                if (munkresStep2(star_graph, cover_table)) {
                    done = true;
                    break;
                }
            case 3:
                if (!munkresStep3(cost_graph, M, star_graph, prime_graph, cover_table, p,
                                  nrows, ncols)) {
                    step = 5;
                    break;
                }
            case 4:
                munkresStep4(star_graph, prime_graph, cover_table, p);
                step = 2;
                break;
            case 5:
                munkresStep5(cost_graph, M, cover_table, nrows, ncols);
                step = 3;
                break;
        }
    }
}

std::size_t assignment_out_workspace(const int M) {
    return sizeof(float) * M * M;
}

void assignment_out(int *connections,         // 2xM
                    const float *score_graph, // MxM
                    const int count_a, const int count_b, const int M,
                    const float score_threshold, void *workspace) {
    const int nrows = count_a;
    const int ncols = count_b;

    // compute cost graph (negate score graph)
    float *cost_graph = (float *) workspace;
    for (int i = 0; i < count_a; i++) {
        for (int j = 0; j < count_b; j++) {
            const int idx = i * M + j;
            cost_graph[idx] = -score_graph[idx];
        }
    }

    // run munkres algorithm
    auto star_graph = PairGraph(nrows, ncols);
    _munkres(cost_graph, M, star_graph, nrows, ncols);

    // fill output connections
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            if (star_graph.isPair(i, j) && score_graph[i * M + j] > score_threshold) {
                connections[0 * M + i] = j;
                connections[1 * M + j] = i;
            }
        }
    }
}


void assignement(const float *score_graph, const int *topology, const int *counts, int *connections,
                 const int K = 21, const int M = 30, const float score_threshold = 0.1) {
    void *workspace = (void *) malloc(assignment_out_workspace(M));
    for (int k = 0; k < K; k++) {
        const int *tk = &topology[k * 4];
        const int cmap_idx_a = tk[2];
        const int cmap_idx_b = tk[3];
        const int count_a = counts[cmap_idx_a];
        const int count_b = counts[cmap_idx_b];
        assignment_out(&connections[k * 2 * M], &score_graph[k * M * M], count_a,
                       count_b, M, score_threshold, workspace);
    }

    free(workspace);
}

//---------------------计算connect-parts-------------------//


std::size_t connect_parts_out_workspace(const int C, const int M) {
    return sizeof(int) * C * M;
}

void connect_parts_out(int *object_counts,     // 1
                       int *objects,           // PxC
                       const int *connections, // Kx2xM
                       const int *topology,    // Kx4
                       const int *counts,      // C
                       const int K, const int C, const int M, const int P,
                       void *workspace) {

    // initialize objects
    for (int i = 0; i < C * M; i++) {
        objects[i] = -1;
    }

    // initialize visited
    std::memset(workspace, 0, connect_parts_out_workspace(C, M));
    int *visited = (int *) workspace;

    int num_objects = 0;

    for (int c = 0; c < C; c++) {
        if (num_objects >= P) {
            break;
        }

        const int count = counts[c];

        for (int i = 0; i < count; i++) {
            if (num_objects >= P) {
                break;
            }

            std::queue<std::pair<int, int>> q;
            bool new_object = false;
            q.push({c, i});

            while (!q.empty()) {
                auto node = q.front();
                q.pop();
                int c_n = node.first;
                int i_n = node.second;

                if (visited[c_n * M + i_n]) {
                    continue;
                }

                visited[c_n * M + i_n] = 1;
                new_object = true;
                objects[num_objects * C + c_n] = i_n;

                for (int k = 0; k < K; k++) {
                    const int *tk = &topology[k * 4];
                    const int c_a = tk[2];
                    const int c_b = tk[3];
                    const int *ck = &connections[k * 2 * M];

                    if (c_a == c_n) {
                        int i_b = ck[i_n];
                        if (i_b >= 0) {
                            q.push({c_b, i_b});
                        }
                    }

                    if (c_b == c_n) {
                        int i_a = ck[M + i_n];
                        if (i_a >= 0) {
                            q.push({c_a, i_a});
                        }
                    }
                }
            }

            if (new_object) {
                num_objects++;
            }
        }
    }
    *object_counts = num_objects;
}

void connect_parts(int *connections, const int *topology, const int *counts, int *object_counts, int *objects,
                   const int K = 21, const int C = 18, const int M = 30, const int P = 30) {
    void *workspace = malloc(connect_parts_out_workspace(C, M));
    connect_parts_out(object_counts, objects, connections, topology, counts, K, C, M, P, workspace);
    free(workspace);
}

#endif //TRTPOSE_POST_PROCESS_H
