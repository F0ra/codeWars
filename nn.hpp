#ifndef NN_MODULE_HPP_
#define NN_MODULE_HPP_

#include <array>
#include <vector>
#include <cassert>
#include <iostream>
#include <algorithm>


#pragma GCC target("avx")  //Enable AVX
#include "align.hpp"
#include <x86intrin.h> //AVX/SSE Extensions
#include <immintrin.h> // For AVX instructions

using std::array;
using std::vector;

namespace nn {

// getters used as arr placeholders for conv2d proc
auto get_3x3_flatArray() {
    static float flat_3x3[3*3];
    return flat_3x3;
}

auto get_5x5_flatArray() {
    static float flat_5x5[5*5];
    return flat_5x5;
}

struct Shape4d {
        int dim_0;
        int dim_1;
        int dim_2;
        int dim_3;
};

struct Shape3d {
        int dim_0;
        int dim_1;
        int dim_2;
};

struct Shape2d {
        int dim_0;
        int dim_1;
};
// 2d tensor for fast matMul operations using avx instructions;
// dim0 and dim1 are transposed in matMul;
typedef vector<vector<float, alignocator<float,32>>> avxTensor2d;

struct tensor2d {
    tensor2d(Shape2d shape)
        : m_dim0(shape.dim_0), m_dim1(shape.dim_1) 
    {
        m_data = new float[m_dim0*m_dim1];
        objCounter = new int;
        *objCounter = 1;      
    }

    tensor2d(Shape3d shape, float *data, int *objCntr)
        : m_dim0(shape.dim_0), m_dim1(shape.dim_1)
    {
        m_data = data;
        objCounter = objCntr;
        *objCounter += 1;
    }

    ~tensor2d() {
        *objCounter -= 1;
        if(*objCounter == 0) {
            delete[] m_data;
        }
    }

    tensor2d& operator + (vector<float> const bias) {
        assert(m_dim1 == bias.size());
        for (int row{}; row < m_dim0; ++row) {
            for (int col{}; col < m_dim1; ++col) {
                m_data[row*m_dim1 + col] += bias[col];
            }
        }
        return *this;
    }    

    std::string shape() {
        std::string shape_;
                    shape_ = "["  + std::to_string(m_dim0);
                    shape_+= ", " + std::to_string(m_dim1);
                    shape_+= "]";
        return shape_;
    };

    int m_dim0, m_dim1;
    float *m_data;
    int *objCounter{nullptr};

    float* getDataPointer() { return m_data;}
};

std::ostream & operator << (std::ostream &out, const tensor2d &t)
{
    for ( int i{}; i < t.m_dim0; ++i) {
        out<<"\n";
        for (int j{}; j < t.m_dim1; ++j) {
            out << " " << t.m_data[i*t.m_dim1 + j];
        }
    }
    out<<"\n";
    return out;
}

struct tensor3d {
 public:
    tensor3d(Shape3d shape)
        : m_dim0(shape.dim_0), m_dim1(shape.dim_1), m_dim2(shape.dim_2) 
    {
        m_data = new float[m_dim0*m_dim1*m_dim2];
        objCounter = new int;
        *objCounter = 1;
    }

    tensor3d(Shape3d shape, float *data, int *objCntr)
        : m_dim0(shape.dim_0), m_dim1(shape.dim_1), m_dim2(shape.dim_2) 
    {
        m_data = data;
        objCounter = objCntr;
        *objCounter += 1;
    }

    ~tensor3d() {
        *objCounter -= 1;
        if(*objCounter == 0) {
            delete[] m_data;
        }
    }

    std::string shape() {
        std::string shape_;
                    shape_ = "["  + std::to_string(m_dim0);
                    shape_+= ", " + std::to_string(m_dim1);
                    shape_+= ", " + std::to_string(m_dim2);
                    shape_+= "]";
        return shape_;
    };

    tensor2d operator[](int index){
        if (index >= m_dim0 || index < 0) throw std::out_of_range ("dim_0 index is out of range");
        return tensor2d({m_dim1,m_dim2}, &m_data[index*m_dim1*m_dim2],objCounter);
    };

    int m_dim0, m_dim1, m_dim2;
    float *m_data;
    int *objCounter{nullptr};

    float* getDataPointer() { return m_data;}
};

std::ostream & operator << (std::ostream &out, const tensor3d &t)
{
    for (int dim_0_index{}; dim_0_index < t.m_dim0; ++dim_0_index) {
            out<<"\n-----------------";
        for (int dim_1_index{}; dim_1_index < t.m_dim1; ++ dim_1_index) {
                out << "\n";
            for (int dim_2_index{}; dim_2_index < t.m_dim2; ++dim_2_index) {
                    out<<" "<<t.m_data[   dim_0_index*t.m_dim1*t.m_dim2 
                                        + dim_1_index*t.m_dim2
                                        + dim_2_index];
            }
        }
    }
    out<<"\n=============\n";
    return out;
}

// tensor4d stores all values in single continuous array
struct tensor4d {

    tensor4d(Shape4d shape)
        : m_dim0(shape.dim_0), m_dim1(shape.dim_1), m_dim2(shape.dim_2), m_dim3(shape.dim_3)
    {
        m_data = new float[m_dim0*m_dim1*m_dim2*m_dim3];
        objCounter = new int;
        *objCounter = 1;
    }


    ~tensor4d() {
        *objCounter -= 1;
        if(*objCounter == 0) {
            delete[] m_data;
        }
    }

    // tensor4d(const tensor4d &t) = delete;
    tensor4d & operator =(const tensor4d ) = delete;

    tensor3d operator[](int index){
        if (index >= m_dim0 || index < 0) throw std::out_of_range ("dim_0 index is out of range");
        return tensor3d({m_dim1,m_dim2,m_dim3}, &m_data[index*m_dim1*m_dim2*m_dim3], objCounter);
    };

    tensor4d& operator + (vector<float> const bias) {
        assert(m_dim1 == bias.size());
        for (int n_sample{}; n_sample < m_dim0; ++n_sample) {
            for (int channel{}; channel < m_dim1; ++channel) {
                for (int row{}; row < m_dim2; ++row) {
                    for (int col{}; col < m_dim3; ++col) {
                        m_data[n_sample*m_dim1*m_dim2*m_dim3 + channel*m_dim2*m_dim3 + row*m_dim3 + col] += bias[channel];
                    }
                }
            }
        }
        return *this;
    }

    // return flat tensor with shape[n_samples][channels*rows*cols]
    tensor2d makeFlat() {
        return tensor2d({m_dim0,m_dim1*m_dim2*m_dim3,}, &m_data[0], objCounter);
    }

    std::string shape() {
        std::string shape_;
                    shape_ = "["  + std::to_string(m_dim0);
                    shape_+= ", " + std::to_string(m_dim1);
                    shape_+= ", " + std::to_string(m_dim2);
                    shape_+= ", " + std::to_string(m_dim3);
                    shape_+= "]";
        return shape_;
    };

    int m_dim0, m_dim1, m_dim2, m_dim3;
    float *m_data;
    int *objCounter{nullptr};
    float* getDataPointer() { return m_data;}
};

std::ostream & operator << (std::ostream &out, const tensor4d &t)
{
    for (int dim_0_index{}; dim_0_index < t.m_dim0; ++dim_0_index) {
            out<<"\n-----------------";

        for (int dim_1_index{}; dim_1_index < t.m_dim1; ++ dim_1_index) {
                out << "\n-----------";
            for (int dim_2_index{}; dim_2_index < t.m_dim2; ++dim_2_index) {
                out << "\n";
                for (int dim_3_index{}; dim_3_index < t.m_dim3; ++dim_3_index) {
                    out<<" "<<t.m_data[  dim_0_index*t.m_dim1*t.m_dim2*t.m_dim3 
                                                        + dim_1_index*t.m_dim2*t.m_dim3
                                                        + dim_2_index*t.m_dim3
                                                        + dim_3_index];
                }
            }
        }
    }
    out<<"\n=============\n";
    return out;
}

void fillTensor4d(tensor4d &in, float val) {
    for (int sample{}; sample < in.m_dim0; ++sample) {
        for (int channel{}; channel < in.m_dim1; ++ channel) {
            for (int row{}; row < in.m_dim2; ++row) {
                for (int col{}; col < in.m_dim3; ++col) {
                    in.m_data[sample*in.m_dim1*in.m_dim2*in.m_dim3 + channel*in.m_dim2*in.m_dim3 + row*in.m_dim3 + col] = val;
                }
            }
        }
    }
}

// count total dot product call
// for estimating work done by convolution layer
int counter{};

float dot(float *vec1, const int vec1Size, float *vec2, const int vec2Size) {
    assert(vec1Size == vec2Size);
    ++counter;

    float dotProduct{};
    for (int i{}; i < vec2Size; ++i) {
        dotProduct += vec1[i] * vec2[i];
    }

    return dotProduct;
}

// avx version of dot product allows process 8 floats at ones
// first vector have to be aligned 32 
float dot_avx(float *a_aligned_32, int size_a, float *b, int size_b) {
    assert( size_a == size_b );
    
    int parallelStep{};
    int parallel = size_a/8;
    int linearStep{};
    int linear = size_a%8;

    float rez{};

        // mask 241 ==> 1000 1000 low bits
        // mask 242 ==> 0100 0100
        // mask 244 ==> 0010 0010
        // mask 248 ==> 0001 0001

    auto process_8_floats = [&] (float *vector1, float *vector2)->float {
        __m256 v1 = _mm256_load_ps(&a_aligned_32[0]);
        __m256 v2 = _mm256_load_ps(&b[0]);
        __m256 v3 = _mm256_dp_ps(v1, v2, 241);
        // return v3[0] + v3[4];

        __m256 v4 = _mm256_hadd_ps(v3, v3);
        __m256 v5 = _mm256_hadd_ps(v4, v4);

        __m128 t1 = _mm256_extractf128_ps(v5, 1);
        __m128 t2 = _mm_add_ss(_mm256_castps256_ps128(v5), t1);
        return _mm_cvtss_f32(t2);
    };

    for ( ; parallelStep < parallel; ++parallelStep) {
        rez += process_8_floats(&a_aligned_32[parallelStep*8], &b[parallelStep*8]);
    }

    for ( ; linearStep < linear; ++linearStep) {

        rez += a_aligned_32[parallelStep*8 + linearStep] * b[parallelStep*8 + linearStep];
    }

    return rez;
}


void matMul(tensor2d &mat1, tensor2d &mat2, tensor2d &out) {
    assert(out.m_dim0 == mat1.m_dim0 && out.m_dim1 == mat2.m_dim1);
    vector<float> tmp(mat2.m_dim0);
    
    auto getMat2ColVector = [&] (int col)->void {
        for (int row{}; row < mat2.m_dim0; ++row) {
            tmp[row] = mat2.m_data[row*mat2.m_dim1 + col];
        }
    };

    for (int row{}; row < out.m_dim0; ++row) {
        for (int col{}; col < out.m_dim1; ++col) {
            getMat2ColVector(col);
            out.m_data[row*out.m_dim1 + col] = dot(&mat1.m_data[row*mat1.m_dim1], mat1.m_dim1, &tmp[0], tmp.size());
        }
    }
}

// dim0 and dim1 are transposed in matMul;
void matMul(tensor2d &mat1, avxTensor2d &mat2, tensor2d &out) {
    assert(out.m_dim0 == mat1.m_dim0 && out.m_dim1 == mat2.size());

    for (int row{}; row < out.m_dim0; ++row) {
        for (int col{}; col < out.m_dim1; ++col) {
            // out.m_data[row*out.m_dim1 + col] = dot_avx(&mat2[col][0],mat2[0].size(),&mat1.m_data[row*mat1.m_dim1], mat1.m_dim1);
            out.m_data[row*out.m_dim1 + col] = dot(&mat2[col][0],mat2[0].size(),&mat1.m_data[row*mat1.m_dim1], mat1.m_dim1);            
        }
    }
}

//  stride 1, padding SAME
void conv2d_old(float *input, const int inputRows, float *filter, const int filterSize, float *out) {
    // input  :  [rows][cols] , have to be square,  inputRows = inputCols and 
    //                          need to be flatten for convolution proc
    // filter :  [rows][cols] , have to be square,  filterSize = filterRows = filterCols 
    //                          filter already comes flat
    // out    :  [rows][cols] , same shape as input
    //  filter shape and input shape must be square, at least now.
    assert(filterSize%2 != 0);  // filter size must be odd

    const int rows = inputRows;
    const int cols = inputRows;
    int row,col;

    //flatten filter
    float *flatFilter = filter;

    auto flattenInputReceptiveField = [&] (const int inputRow, const int inputCol, float *tmp) {
        const int filterAreaRowStart = inputRow - filterSize/2;
        const int filterAreaColStart = inputCol - filterSize/2;
        for (int i{}; i < filterSize; ++i) {
            for (int j{}; j < filterSize; ++j) {
                // checking boundaries , if out - assign 0
                if ((filterAreaRowStart + i) < 0 || (filterAreaRowStart + i) >= rows ||
                    (filterAreaColStart + j) < 0 || (filterAreaColStart + j) >= cols ) {
                    tmp[i*filterSize + j] = 0;
                } else {
                    tmp[i*filterSize + j] = input[(filterAreaRowStart + i)* inputRows
                                                            + (filterAreaColStart + j)];
                }
            }
        }
    };

    auto tmp = (filterSize == 5) ? get_5x5_flatArray() : get_3x3_flatArray();

    for (row = 0; row < rows; ++row) {
        for (col = 0; col < cols; ++col) {
                flattenInputReceptiveField(row, col, tmp);
                out[row*cols + col] += dot(tmp, filterSize*filterSize, 
                                                flatFilter, filterSize*filterSize);
        }
    }
}


//  stride 1, padding SAME
void conv2d(float *input, const int inputRows, float *filter, const int filterSize, float *out) {
    // input  :  [rows][cols] , have to be square,  inputRows = inputCols and 
    //                          need to be flatten for convolution proc
    // filter :  [rows][cols] , have to be square,  filterSize = filterRows = filterCols 
    //                          filter already comes flat
    // out    :  [rows][cols] , same shape as input
    //  filter shape and input shape must be square, at least now.
    assert(filterSize%2 != 0);  // filter size must be odd
    assert(filterSize == 3 || filterSize == 5);  // only filtersize = (3 or 5) implemented

    const int rows = inputRows;
    const int cols = inputRows;
    const int zeroPaddingSize = filterSize/2;
    int row,col;

    //flatten filter
    float *flatFilter = filter;

    const int zeroPaddingRows = inputRows + 2*zeroPaddingSize;
    const int zeroPaddingCols = inputRows + 2*zeroPaddingSize;

    vector<float> zeroPaddingInput(zeroPaddingRows * zeroPaddingCols);
    for (int row{}; row < zeroPaddingRows; ++row) {
        for (int col{}; col < zeroPaddingCols; ++col) {
            if (row < zeroPaddingSize || row >= zeroPaddingRows - zeroPaddingSize ||
                col < zeroPaddingSize || col >= zeroPaddingCols - zeroPaddingSize) {
                    zeroPaddingInput[row*zeroPaddingCols + col] = 0;
                } else {
                    zeroPaddingInput[row*zeroPaddingCols + col] = input[(row-zeroPaddingSize)*cols + (col-zeroPaddingSize)];
                }
        }
    }

    auto tmp = (filterSize == 5) ? get_5x5_flatArray() : get_3x3_flatArray();

    if (filterSize == 3) {
        for (row = 0; row < rows; ++row) {
            for (col = 0; col < cols; ++col) {
                out[row*cols + col] +=  dot(&zeroPaddingInput[(row)*zeroPaddingCols + col], filterSize, &flatFilter[0], filterSize)+
                                        dot(&zeroPaddingInput[(row+1)*zeroPaddingCols + col], filterSize, &flatFilter[3], filterSize)+
                                        dot(&zeroPaddingInput[(row+2)*zeroPaddingCols + col], filterSize, &flatFilter[6], filterSize);
            }
        }
    } 
    
    if (filterSize == 5) {
        for (row = 0; row < rows; ++row) {
            for (col = 0; col < cols; ++col) {
                out[row*cols + col] +=  dot(&zeroPaddingInput[(row)*zeroPaddingCols + col], filterSize, &flatFilter[0], filterSize)+
                                        dot(&zeroPaddingInput[(row+1)*zeroPaddingCols + col], filterSize, &flatFilter[5], filterSize)+
                                        dot(&zeroPaddingInput[(row+2)*zeroPaddingCols + col], filterSize, &flatFilter[10], filterSize)+
                                        dot(&zeroPaddingInput[(row+3)*zeroPaddingCols + col], filterSize, &flatFilter[15], filterSize)+
                                        dot(&zeroPaddingInput[(row+4)*zeroPaddingCols + col], filterSize, &flatFilter[20], filterSize);
            }
        }
    } 
}


tensor4d convolution(tensor4d &in, tensor4d &filter) {
    // in tensor4d : [n samples][rows][cols][in channels];
    // out tensor4d : [n samples][rows][cols][out channels];
    // filter tensor4d : [out channels][in channels][filter rows][filter cols]
    tensor4d out({in.m_dim0, filter.m_dim0, in.m_dim2, in.m_dim3});
    
    for (int sample{}; sample < in.m_dim0; ++sample) {
        for (int outChannel{}; outChannel < filter.m_dim0; ++outChannel) {
            for (int inChannel{}; inChannel < filter.m_dim1; ++inChannel) {
                int filterstartIndex = outChannel*filter.m_dim1*filter.m_dim2*filter.m_dim3 + inChannel*filter.m_dim2*filter.m_dim3 ;
                int inputStartIndex =  sample*in.m_dim1*in.m_dim2*in.m_dim3 + inChannel*in.m_dim2*in.m_dim3;
                int outputStartIndex = sample*filter.m_dim0*in.m_dim2*in.m_dim3 + outChannel*in.m_dim2*in.m_dim3;

                conv2d(&in.m_data[inputStartIndex], in.m_dim2, &filter.m_data[filterstartIndex],filter.m_dim2, &out.m_data[outputStartIndex]);
            }
        }
    }
        
    return out;
}

void convolution(tensor4d &in, tensor4d &filter, tensor4d &out) {
    // in tensor4d : [n samples][in channels][rows][cols];
    // out tensor4d : [n samples][out channels][rows][cols];
    // filter tensor4d : [out channels][in channels][filter rows][filter cols]
    assert(in.m_dim0==out.m_dim0 && filter.m_dim0==out.m_dim1 && in.m_dim2==out.m_dim2 && in.m_dim3==out.m_dim3);
    for (int sample{}; sample < in.m_dim0; ++sample) {
        for (int outChannel{}; outChannel < filter.m_dim0; ++outChannel) {
            for (int inChannel{}; inChannel < filter.m_dim1; ++inChannel) {
                int filterstartIndex = outChannel*filter.m_dim1*filter.m_dim2*filter.m_dim3 + inChannel*filter.m_dim2*filter.m_dim3 ;
                int inputStartIndex =  sample*in.m_dim1*in.m_dim2*in.m_dim3 + inChannel*in.m_dim2*in.m_dim3;
                int outputStartIndex = sample*filter.m_dim0*in.m_dim2*in.m_dim3 + outChannel*in.m_dim2*in.m_dim3;

                conv2d(&in.m_data[inputStartIndex], in.m_dim2, &filter.m_data[filterstartIndex],filter.m_dim2, &out.m_data[outputStartIndex]);
            }
        }
    }
}

// activation func
float reLu(float activation) {
    return (activation < 0) ? 0 : activation;
}

void reLu(tensor4d &t) {
    for (int n_sample{}; n_sample < t.m_dim0; ++n_sample) {
        for (int channel{}; channel < t.m_dim1; ++channel) {
            for (int row{}; row < t.m_dim2; ++row) {
                for (int col{}; col < t.m_dim3; ++col) {
                    t.m_data[n_sample*t.m_dim1*t.m_dim2*t.m_dim3 + channel*t.m_dim2*t.m_dim3 + row*t.m_dim3 + col] = 
                    reLu(t.m_data[n_sample*t.m_dim1*t.m_dim2*t.m_dim3 + channel*t.m_dim2*t.m_dim3 + row*t.m_dim3 + col]);
                }
            }
        }
    }
}

void reLu(tensor2d &t) {
    for (int row{}; row < t.m_dim0; ++row) {
        for (int col{}; col < t.m_dim1; ++col) {
            t.m_data[row*t.m_dim1 + col] =  reLu(t.m_data[row*t.m_dim1 + col]);
        }
    }
}

void maxPull(tensor4d &in, tensor4d &out) {
    // in tensor4d : [n samples][in channels][rows][cols];
    // out tensor4d : [n samples][out channels][rows/2][cols/2];
    assert( in.m_dim0==out.m_dim0 &&
            in.m_dim1==out.m_dim1 && 
            in.m_dim2/2==out.m_dim2 && 
            in.m_dim3/2==out.m_dim3);
    
    vector<float> tmp(4);

    for (int n_sample{}; n_sample < in.m_dim0; ++n_sample) {
        for (int channel{}; channel < in.m_dim1; ++channel) {
            for (int row{}; row < out.m_dim2; ++row) {
                for (int col{}; col < out.m_dim3; ++col) {
                    tmp[0] = in.m_data[n_sample*in.m_dim1*in.m_dim2*in.m_dim3 + channel*in.m_dim2*in.m_dim3 + (row*2)*in.m_dim3 + col*2];
                    tmp[1] = in.m_data[n_sample*in.m_dim1*in.m_dim2*in.m_dim3 + channel*in.m_dim2*in.m_dim3 + (row*2)*in.m_dim3 + col*2 + 1];
                    tmp[2] = in.m_data[n_sample*in.m_dim1*in.m_dim2*in.m_dim3 + channel*in.m_dim2*in.m_dim3 + (row*2+1)*in.m_dim3 + col*2];
                    tmp[3] = in.m_data[n_sample*in.m_dim1*in.m_dim2*in.m_dim3 + channel*in.m_dim2*in.m_dim3 + (row*2+1)*in.m_dim3 + col*2 + 1];
                    out.m_data[n_sample*in.m_dim1*out.m_dim2*out.m_dim3 + channel*out.m_dim2*out.m_dim3 + (row)*out.m_dim3 + col] = *std::max_element(tmp.begin(), tmp.end());
                }
            }
        }
    }
}

tensor2d softMax(tensor2d &t) {
    auto softmax = [&] (float *arr, int size, float *soft) {
        float sum{};
        for (int i{}; i < size; ++i) {
            sum+=std::pow(M_E, arr[i]);
        }
        for (int i{}; i < size; ++i) {
            soft[i] = std::pow(M_E, arr[i])/sum;
        }   
    };

    tensor2d out({t.m_dim0, t.m_dim1});
    for (int row{}; row < t.m_dim0; ++row) {
        softmax(&t.m_data[row*t.m_dim1], t.m_dim1, &out.m_data[row*t.m_dim1]);
    }
    return out;
}

vector<int> argMax(tensor2d &t) {
    vector<int> out(t.m_dim0);
    float maxVal{};
    int maxIndex{};
    for (int row{}; row < t.m_dim0; ++row) {
        maxVal = t.m_data[row*t.m_dim1];
        maxIndex = 0;
        for (int col{}; col < t.m_dim1; ++col) {
            if(t.m_data[row*t.m_dim1 + col] > maxVal) {
                maxVal = t.m_data[row*t.m_dim1 + col];
                maxIndex = col;
            }
        }
        out[row] = maxIndex;
    }
    return out;
}

void reshapeTensor_ChannelRowsCols_to_RowsColsChannel(tensor4d &t) {
    int n_samples = t.m_dim0;
    int channels  = t.m_dim1;
    int rows      = t.m_dim2;
    int cols      = t.m_dim3;
    int size      = t.m_dim0*t.m_dim1*t.m_dim2*t.m_dim3;
    int index{};
    vector<float> tmp(size);
    for (int i{}; i<size; ++i) {
        tmp[i] = *(t.m_data+i);
    }

    for (int sample{}; sample < n_samples; ++sample) {
        for (int row{}; row < rows; ++row) {
            for (int col{}; col < cols; ++col) {
                for (int channel{}; channel < channels; ++channel) {
                    t.m_data[index] = tmp[sample*rows*cols*channels + channel*rows*cols + row*cols + col];
                    ++index;
                }
            }
        }
    }
    
    t.m_dim1 = rows;
    t.m_dim2 = cols;
    t.m_dim3 = channels;
}

nn::tensor4d extractFilter4dFromArr(Shape4d shape, float *data) {
    //  extracted filter shape: [out channels][in channels][filter rows][filter cols]
    int dim0, dim1, dim2, dim3;
    dim0 = shape.dim_0;
    dim1 = shape.dim_1;
    dim2 = shape.dim_2;
    dim3 = shape.dim_3;

    nn::tensor4d out({dim3, dim2, dim0, dim1});

    int data_index{};

    for (int dim3_i{}; dim3_i < dim3; ++dim3_i) {
        for (int dim2_i{}; dim2_i < dim2; ++dim2_i) {
            for (int dim0_i{}; dim0_i < dim0; ++dim0_i) {
                for (int dim1_i{}; dim1_i < dim1; ++dim1_i) {
                    out.m_data[data_index] = data[dim0_i*dim1*dim2*dim3 + dim1_i*dim2*dim3 + dim2_i*dim3 + dim3_i];
                    ++data_index;
                }
            }
        }
    }
    return out;
}

nn::tensor4d extractInput4dFromArr(Shape4d shape, float *data) {
    //  extracted input shape [n samples][out channels][rows][cols];
    int dim0, dim1, dim2, dim3;
    dim0 = shape.dim_0;
    dim1 = shape.dim_1;
    dim2 = shape.dim_2;
    dim3 = shape.dim_3;

    nn::tensor4d out({dim0, dim3, dim1, dim2});

    int data_index{};

    for (int dim0_i{}; dim0_i < dim0; ++dim0_i) {
        for (int dim3_i{}; dim3_i < dim3; ++dim3_i) {
            for (int dim1_i{}; dim1_i < dim1; ++dim1_i) {
                for (int dim2_i{}; dim2_i < dim2; ++dim2_i) {
                    out.m_data[data_index] = data[dim0_i*dim1*dim2*dim3 + dim1_i*dim2*dim3 + dim2_i*dim3 + dim3_i];
                    ++data_index;
                }
            }
        }
    }
    return out;
}

};

#endif  // NN_MODULE_HPP_