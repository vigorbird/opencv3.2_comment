/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2009, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

/** Authors: Ethan Rublee, Vincent Rabaud, Gary Bradski */

#include "precomp.hpp"
#include "opencl_kernels_features2d.hpp"
#include <iterator>

#ifndef CV_IMPL_ADD
#define CV_IMPL_ADD(x)
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cv
{

const float HARRIS_K = 0.04f;

template<typename _Tp> inline void copyVectorToUMat(const std::vector<_Tp>& v, OutputArray um)
{
    if(v.empty())
        um.release();
    else
        Mat(1, (int)(v.size()*sizeof(v[0])), CV_8U, (void*)&v[0]).copyTo(um);
}

#ifdef HAVE_OPENCL
static bool
ocl_HarrisResponses(const UMat& imgbuf,
                    const UMat& layerinfo,
                    const UMat& keypoints,
                    UMat& responses,
                    int nkeypoints, int blockSize, float harris_k)
{
    size_t globalSize[] = {(size_t)nkeypoints};

    float scale = 1.f/((1 << 2) * blockSize * 255.f);
    float scale_sq_sq = scale * scale * scale * scale;

    ocl::Kernel hr_ker("ORB_HarrisResponses", ocl::features2d::orb_oclsrc,
                format("-D ORB_RESPONSES -D blockSize=%d -D scale_sq_sq=%.12ef -D HARRIS_K=%.12ff", blockSize, scale_sq_sq, harris_k));
    if( hr_ker.empty() )
        return false;

    return hr_ker.args(ocl::KernelArg::ReadOnlyNoSize(imgbuf),
                ocl::KernelArg::PtrReadOnly(layerinfo),
                ocl::KernelArg::PtrReadOnly(keypoints),
                ocl::KernelArg::PtrWriteOnly(responses),
                nkeypoints).run(1, globalSize, 0, true);
}

static bool
ocl_ICAngles(const UMat& imgbuf, const UMat& layerinfo,
             const UMat& keypoints, UMat& responses,
             const UMat& umax, int nkeypoints, int half_k)
{
    size_t globalSize[] = {(size_t)nkeypoints};

    ocl::Kernel icangle_ker("ORB_ICAngle", ocl::features2d::orb_oclsrc, "-D ORB_ANGLES");
    if( icangle_ker.empty() )
        return false;

    return icangle_ker.args(ocl::KernelArg::ReadOnlyNoSize(imgbuf),
                ocl::KernelArg::PtrReadOnly(layerinfo),
                ocl::KernelArg::PtrReadOnly(keypoints),
                ocl::KernelArg::PtrWriteOnly(responses),
                ocl::KernelArg::PtrReadOnly(umax),
                nkeypoints, half_k).run(1, globalSize, 0, true);
}


static bool
ocl_computeOrbDescriptors(const UMat& imgbuf, const UMat& layerInfo,
                          const UMat& keypoints, UMat& desc, const UMat& pattern,
                          int nkeypoints, int dsize, int wta_k)
{
    size_t globalSize[] = {(size_t)nkeypoints};

    ocl::Kernel desc_ker("ORB_computeDescriptor", ocl::features2d::orb_oclsrc,
                         format("-D ORB_DESCRIPTORS -D WTA_K=%d", wta_k));
    if( desc_ker.empty() )
        return false;

    return desc_ker.args(ocl::KernelArg::ReadOnlyNoSize(imgbuf),
                         ocl::KernelArg::PtrReadOnly(layerInfo),
                         ocl::KernelArg::PtrReadOnly(keypoints),
                         ocl::KernelArg::PtrWriteOnly(desc),
                         ocl::KernelArg::PtrReadOnly(pattern),
                         nkeypoints, dsize).run(1, globalSize, 0, true);
}
#endif

/**
 * Function that computes the Harris responses in a
 * blockSize x blockSize patch at given points in the image
 */
 //img=输入的各个金字塔层下的图像
 //layerinfo=每层图像提取的矩阵
 //pts=检测得到的所有特征点
 //blockSize=7
 //harris_k=0.04
static void
HarrisResponses(const Mat& img, const std::vector<Rect>& layerinfo, std::vector<KeyPoint>& pts, int blockSize, float harris_k)
{
    CV_Assert( img.type() == CV_8UC1 && blockSize*blockSize <= 2048 );

    size_t ptidx, ptsize = pts.size();

    const uchar* ptr00 = img.ptr<uchar>();//金字塔所有图像的起始地址
    int step = (int)(img.step/img.elemSize1());//表示一行之间需要移动多少个地址
    int r = blockSize/2;// 3

    float scale = 1.f/((1 << 2) * blockSize * 255.f);//1/(4*7*255)=
    float scale_sq_sq = scale * scale * scale * scale;

    AutoBuffer<int> ofsbuf(blockSize*blockSize);
    int* ofs = ofsbuf;//这个变量存储的是在一个7*7的正方形中的地址
    for( int i = 0; i < blockSize; i++ )
        for( int j = 0; j < blockSize; j++ )
            ofs[i*blockSize + j] = (int)(i*step + j);

    for( ptidx = 0; ptidx < ptsize; ptidx++ )//遍历所有的特征点
    {
        int x0 = cvRound(pts[ptidx].pt.x);//提取出特征点的像素坐标，注意这里得到的特征点还是在各层金字塔下的坐标并不是在原始图像下的坐标
        int y0 = cvRound(pts[ptidx].pt.y);
        int z = pts[ptidx].octave;
	//在这个特征点周围构造一个正方形，这个地址就是这个正方形左上角的地址
        const uchar* ptr0 = ptr00 + (y0 - r + layerinfo[z].y)*step + x0 - r + layerinfo[z].x;
        int a = 0, b = 0, c = 0;

        for( int k = 0; k < blockSize*blockSize; k++ )//遍历这个正方形中的所有像素
        {
            const uchar* ptr = ptr0 + ofs[k];//得到这个正方向像素所对应的地址
            int Ix = (ptr[1] - ptr[-1])*2 + (ptr[-step+1] - ptr[-step-1]) + (ptr[step+1] - ptr[step-1]);
            int Iy = (ptr[step] - ptr[-step])*2 + (ptr[step-1] - ptr[-step-1]) + (ptr[step+1] - ptr[-step+1]);
            a += Ix*Ix;
            b += Iy*Iy;
            c += Ix*Iy;
        }
        pts[ptidx].response = ((float)a * b - (float)c * c - harris_k * ((float)a + b) * ((float)a + b))*scale_sq_sq;//计算得到这个特征点的相应
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//ICAngles函数
//img=输入的金字塔图像
//layerinfo=对应的矩阵块
//pts经过harris筛选过的特征点
//默认值umax=15，15，15，15，14，14，14，13，13，12，11，10，9，8，6，3，0=17维
//half_k=默认值是15
//主要是更新特征点的方向
static void ICAngles(const Mat& img, const std::vector<Rect>& layerinfo, std::vector<KeyPoint>& pts, const std::vector<int> & u_max, int half_k)
{
    int step = (int)img.step1();
    size_t ptidx, ptsize = pts.size();

    for( ptidx = 0; ptidx < ptsize; ptidx++ )
    {
        const Rect& layer = layerinfo[pts[ptidx].octave];
        const uchar* center = &img.at<uchar>(cvRound(pts[ptidx].pt.y) + layer.y, cvRound(pts[ptidx].pt.x) + layer.x);//得到特征点在金字塔图像的地址

        int m_01 = 0, m_10 = 0;

        // Treat the center line differently, v=0
        //m_10=y*I(x,y)的求和
        for (int u = -half_k; u <= half_k; ++u)
            m_10 += u * center[u];

        // Go line by line in the circular patch
        for (int v = 1; v <= half_k; ++v)
        {
            // Proceed over the two lines
            int v_sum = 0;
            int d = u_max[v];
            for (int u = -d; u <= d; ++u)
            {
                int val_plus = center[u + v*step], val_minus = center[u - v*step];
                v_sum += (val_plus - val_minus);
                m_10 += u * (val_plus + val_minus);
            }
            m_01 += v * v_sum;
        }

        pts[ptidx].angle = fastAtan2((float)m_01, (float)m_10);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//computeOrbDescriptors函数
//imagePyramid=输入的金字塔图像
//layerInfo=每一层图像对应的矩阵
//layerScale=内容是缩放的比例即:1，1.2，1.2的平方。。。。1.2的7次方
//keypoints=检测到的特征点
//descriptors=输出的描述子矩阵
//_pattern=输入的进行brief计算的比较像素点
//dsize=输出的描述子维度，默认值是31
//wta_k=表示brief描述子是由几个像素点比较得到，默认值是2
static void
computeOrbDescriptors( const Mat& imagePyramid, const std::vector<Rect>& layerInfo,
                       const std::vector<float>& layerScale, std::vector<KeyPoint>& keypoints,
                       Mat& descriptors, const std::vector<Point>& _pattern, int dsize, int wta_k )
{
    int step = (int)imagePyramid.step;
    int j, i, nkeypoints = (int)keypoints.size();

    for( j = 0; j < nkeypoints; j++ )
    {
        const KeyPoint& kpt = keypoints[j];
        const Rect& layer = layerInfo[kpt.octave];
        float scale = 1.f/layerScale[kpt.octave];
        float angle = kpt.angle;

        angle *= (float)(CV_PI/180.f);//将角度从度变为π表示
        float a = (float)cos(angle), b = (float)sin(angle);

        const uchar* center = &imagePyramid.at<uchar>(cvRound(kpt.pt.y*scale) + layer.y,
                                                      cvRound(kpt.pt.x*scale) + layer.x);//特征点在金字塔图像中的坐标
        float x, y;
        int ix, iy;
        const Point* pattern = &_pattern[0];
        uchar* desc = descriptors.ptr<uchar>(j);
   //详见算法实现文档
    #if 1
        #define GET_VALUE(idx) \
               (x = pattern[idx].x*a - pattern[idx].y*b, \
                y = pattern[idx].x*b + pattern[idx].y*a, \
                ix = cvRound(x), \
                iy = cvRound(y), \
                *(center + iy*step + ix) )//这个函数非常重要!!!!!!!!!!!!!!
     /*我们这里把不需要的代码注释掉方便看代码
    #else
        #define GET_VALUE(idx) \
            (x = pattern[idx].x*a - pattern[idx].y*b, \
            y = pattern[idx].x*b + pattern[idx].y*a, \
            ix = cvFloor(x), iy = cvFloor(y), \
            x -= ix, y -= iy, \
            cvRound(center[iy*step + ix]*(1-x)*(1-y) + center[(iy+1)*step + ix]*(1-x)*y + \
                    center[iy*step + ix+1]*x*(1-y) + center[(iy+1)*step + ix+1]*x*y))
    #endif
    */
        if( wta_k == 2 )
        {
            for (i = 0; i < dsize; ++i, pattern += 16)//遍历
            {
                int t0, t1, val;
                t0 = GET_VALUE(0); t1 = GET_VALUE(1);
                val = t0 < t1;
                t0 = GET_VALUE(2); t1 = GET_VALUE(3);
                val |= (t0 < t1) << 1;
                t0 = GET_VALUE(4); t1 = GET_VALUE(5);
                val |= (t0 < t1) << 2;
                t0 = GET_VALUE(6); t1 = GET_VALUE(7);
                val |= (t0 < t1) << 3;
                t0 = GET_VALUE(8); t1 = GET_VALUE(9);
                val |= (t0 < t1) << 4;
                t0 = GET_VALUE(10); t1 = GET_VALUE(11);
                val |= (t0 < t1) << 5;
                t0 = GET_VALUE(12); t1 = GET_VALUE(13);
                val |= (t0 < t1) << 6;
                t0 = GET_VALUE(14); t1 = GET_VALUE(15);
                val |= (t0 < t1) << 7;

                desc[i] = (uchar)val;//最终得到这个特征点的描述子
            }
        }
	/*
        else if( wta_k == 3 )
        {
            for (i = 0; i < dsize; ++i, pattern += 12)
            {
                int t0, t1, t2, val;
                t0 = GET_VALUE(0); t1 = GET_VALUE(1); t2 = GET_VALUE(2);
                val = t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0);

                t0 = GET_VALUE(3); t1 = GET_VALUE(4); t2 = GET_VALUE(5);
                val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 2;

                t0 = GET_VALUE(6); t1 = GET_VALUE(7); t2 = GET_VALUE(8);
                val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 4;

                t0 = GET_VALUE(9); t1 = GET_VALUE(10); t2 = GET_VALUE(11);
                val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 6;

                desc[i] = (uchar)val;
            }
        }
        else if( wta_k == 4 )
        {
            for (i = 0; i < dsize; ++i, pattern += 16)
            {
                int t0, t1, t2, t3, u, v, k, val;
                t0 = GET_VALUE(0); t1 = GET_VALUE(1);
                t2 = GET_VALUE(2); t3 = GET_VALUE(3);
                u = 0, v = 2;
                if( t1 > t0 ) t0 = t1, u = 1;
                if( t3 > t2 ) t2 = t3, v = 3;
                k = t0 > t2 ? u : v;
                val = k;

                t0 = GET_VALUE(4); t1 = GET_VALUE(5);
                t2 = GET_VALUE(6); t3 = GET_VALUE(7);
                u = 0, v = 2;
                if( t1 > t0 ) t0 = t1, u = 1;
                if( t3 > t2 ) t2 = t3, v = 3;
                k = t0 > t2 ? u : v;
                val |= k << 2;

                t0 = GET_VALUE(8); t1 = GET_VALUE(9);
                t2 = GET_VALUE(10); t3 = GET_VALUE(11);
                u = 0, v = 2;
                if( t1 > t0 ) t0 = t1, u = 1;
                if( t3 > t2 ) t2 = t3, v = 3;
                k = t0 > t2 ? u : v;
                val |= k << 4;

                t0 = GET_VALUE(12); t1 = GET_VALUE(13);
                t2 = GET_VALUE(14); t3 = GET_VALUE(15);
                u = 0, v = 2;
                if( t1 > t0 ) t0 = t1, u = 1;
                if( t3 > t2 ) t2 = t3, v = 3;
                k = t0 > t2 ? u : v;
                val |= k << 6;

                desc[i] = (uchar)val;
            }
        }
        else
            CV_Error( Error::StsBadSize, "Wrong wta_k. It can be only 2, 3 or 4." );
         */
        #undef GET_VALUE
    }
}


static void initializeOrbPattern( const Point* pattern0, std::vector<Point>& pattern, int ntuples, int tupleSize, int poolSize )
{
    RNG rng(0x12345678);
    int i, k, k1;
    pattern.resize(ntuples*tupleSize);

    for( i = 0; i < ntuples; i++ )
    {
        for( k = 0; k < tupleSize; k++ )
        {
            for(;;)
            {
                int idx = rng.uniform(0, poolSize);
                Point pt = pattern0[idx];
                for( k1 = 0; k1 < k; k1++ )
                    if( pattern[tupleSize*i + k1] == pt )
                        break;
                if( k1 == k )
                {
                    pattern[tupleSize*i + k] = pt;
                    break;
                }
            }
        }
    }
}
//这个变量是用于进行brief描述子描述的，此处作者选择使用256对点即512个点
//第一行8,-3, 9,5，表示以特征点为中心将坐标为(8，-3)和(9,5)的像素进行比较
static int bit_pattern_31_[256*4] =
{
    8,-3, 9,5/*mean (0), correlation (0)*/,
    4,2, 7,-12/*mean (1.12461e-05), correlation (0.0437584)*/,
    -11,9, -8,2/*mean (3.37382e-05), correlation (0.0617409)*/,
    7,-12, 12,-13/*mean (5.62303e-05), correlation (0.0636977)*/,
    2,-13, 2,12/*mean (0.000134953), correlation (0.085099)*/,
    1,-7, 1,6/*mean (0.000528565), correlation (0.0857175)*/,
    -2,-10, -2,-4/*mean (0.0188821), correlation (0.0985774)*/,
    -13,-13, -11,-8/*mean (0.0363135), correlation (0.0899616)*/,
    -13,-3, -12,-9/*mean (0.121806), correlation (0.099849)*/,
    10,4, 11,9/*mean (0.122065), correlation (0.093285)*/,
    -13,-8, -8,-9/*mean (0.162787), correlation (0.0942748)*/,
    -11,7, -9,12/*mean (0.21561), correlation (0.0974438)*/,
    7,7, 12,6/*mean (0.160583), correlation (0.130064)*/,
    -4,-5, -3,0/*mean (0.228171), correlation (0.132998)*/,
    -13,2, -12,-3/*mean (0.00997526), correlation (0.145926)*/,
    -9,0, -7,5/*mean (0.198234), correlation (0.143636)*/,
    12,-6, 12,-1/*mean (0.0676226), correlation (0.16689)*/,
    -3,6, -2,12/*mean (0.166847), correlation (0.171682)*/,
    -6,-13, -4,-8/*mean (0.101215), correlation (0.179716)*/,
    11,-13, 12,-8/*mean (0.200641), correlation (0.192279)*/,
    4,7, 5,1/*mean (0.205106), correlation (0.186848)*/,
    5,-3, 10,-3/*mean (0.234908), correlation (0.192319)*/,
    3,-7, 6,12/*mean (0.0709964), correlation (0.210872)*/,
    -8,-7, -6,-2/*mean (0.0939834), correlation (0.212589)*/,
    -2,11, -1,-10/*mean (0.127778), correlation (0.20866)*/,
    -13,12, -8,10/*mean (0.14783), correlation (0.206356)*/,
    -7,3, -5,-3/*mean (0.182141), correlation (0.198942)*/,
    -4,2, -3,7/*mean (0.188237), correlation (0.21384)*/,
    -10,-12, -6,11/*mean (0.14865), correlation (0.23571)*/,
    5,-12, 6,-7/*mean (0.222312), correlation (0.23324)*/,
    5,-6, 7,-1/*mean (0.229082), correlation (0.23389)*/,
    1,0, 4,-5/*mean (0.241577), correlation (0.215286)*/,
    9,11, 11,-13/*mean (0.00338507), correlation (0.251373)*/,
    4,7, 4,12/*mean (0.131005), correlation (0.257622)*/,
    2,-1, 4,4/*mean (0.152755), correlation (0.255205)*/,
    -4,-12, -2,7/*mean (0.182771), correlation (0.244867)*/,
    -8,-5, -7,-10/*mean (0.186898), correlation (0.23901)*/,
    4,11, 9,12/*mean (0.226226), correlation (0.258255)*/,
    0,-8, 1,-13/*mean (0.0897886), correlation (0.274827)*/,
    -13,-2, -8,2/*mean (0.148774), correlation (0.28065)*/,
    -3,-2, -2,3/*mean (0.153048), correlation (0.283063)*/,
    -6,9, -4,-9/*mean (0.169523), correlation (0.278248)*/,
    8,12, 10,7/*mean (0.225337), correlation (0.282851)*/,
    0,9, 1,3/*mean (0.226687), correlation (0.278734)*/,
    7,-5, 11,-10/*mean (0.00693882), correlation (0.305161)*/,
    -13,-6, -11,0/*mean (0.0227283), correlation (0.300181)*/,
    10,7, 12,1/*mean (0.125517), correlation (0.31089)*/,
    -6,-3, -6,12/*mean (0.131748), correlation (0.312779)*/,
    10,-9, 12,-4/*mean (0.144827), correlation (0.292797)*/,
    -13,8, -8,-12/*mean (0.149202), correlation (0.308918)*/,
    -13,0, -8,-4/*mean (0.160909), correlation (0.310013)*/,
    3,3, 7,8/*mean (0.177755), correlation (0.309394)*/,
    5,7, 10,-7/*mean (0.212337), correlation (0.310315)*/,
    -1,7, 1,-12/*mean (0.214429), correlation (0.311933)*/,
    3,-10, 5,6/*mean (0.235807), correlation (0.313104)*/,
    2,-4, 3,-10/*mean (0.00494827), correlation (0.344948)*/,
    -13,0, -13,5/*mean (0.0549145), correlation (0.344675)*/,
    -13,-7, -12,12/*mean (0.103385), correlation (0.342715)*/,
    -13,3, -11,8/*mean (0.134222), correlation (0.322922)*/,
    -7,12, -4,7/*mean (0.153284), correlation (0.337061)*/,
    6,-10, 12,8/*mean (0.154881), correlation (0.329257)*/,
    -9,-1, -7,-6/*mean (0.200967), correlation (0.33312)*/,
    -2,-5, 0,12/*mean (0.201518), correlation (0.340635)*/,
    -12,5, -7,5/*mean (0.207805), correlation (0.335631)*/,
    3,-10, 8,-13/*mean (0.224438), correlation (0.34504)*/,
    -7,-7, -4,5/*mean (0.239361), correlation (0.338053)*/,
    -3,-2, -1,-7/*mean (0.240744), correlation (0.344322)*/,
    2,9, 5,-11/*mean (0.242949), correlation (0.34145)*/,
    -11,-13, -5,-13/*mean (0.244028), correlation (0.336861)*/,
    -1,6, 0,-1/*mean (0.247571), correlation (0.343684)*/,
    5,-3, 5,2/*mean (0.000697256), correlation (0.357265)*/,
    -4,-13, -4,12/*mean (0.00213675), correlation (0.373827)*/,
    -9,-6, -9,6/*mean (0.0126856), correlation (0.373938)*/,
    -12,-10, -8,-4/*mean (0.0152497), correlation (0.364237)*/,
    10,2, 12,-3/*mean (0.0299933), correlation (0.345292)*/,
    7,12, 12,12/*mean (0.0307242), correlation (0.366299)*/,
    -7,-13, -6,5/*mean (0.0534975), correlation (0.368357)*/,
    -4,9, -3,4/*mean (0.099865), correlation (0.372276)*/,
    7,-1, 12,2/*mean (0.117083), correlation (0.364529)*/,
    -7,6, -5,1/*mean (0.126125), correlation (0.369606)*/,
    -13,11, -12,5/*mean (0.130364), correlation (0.358502)*/,
    -3,7, -2,-6/*mean (0.131691), correlation (0.375531)*/,
    7,-8, 12,-7/*mean (0.160166), correlation (0.379508)*/,
    -13,-7, -11,-12/*mean (0.167848), correlation (0.353343)*/,
    1,-3, 12,12/*mean (0.183378), correlation (0.371916)*/,
    2,-6, 3,0/*mean (0.228711), correlation (0.371761)*/,
    -4,3, -2,-13/*mean (0.247211), correlation (0.364063)*/,
    -1,-13, 1,9/*mean (0.249325), correlation (0.378139)*/,
    7,1, 8,-6/*mean (0.000652272), correlation (0.411682)*/,
    1,-1, 3,12/*mean (0.00248538), correlation (0.392988)*/,
    9,1, 12,6/*mean (0.0206815), correlation (0.386106)*/,
    -1,-9, -1,3/*mean (0.0364485), correlation (0.410752)*/,
    -13,-13, -10,5/*mean (0.0376068), correlation (0.398374)*/,
    7,7, 10,12/*mean (0.0424202), correlation (0.405663)*/,
    12,-5, 12,9/*mean (0.0942645), correlation (0.410422)*/,
    6,3, 7,11/*mean (0.1074), correlation (0.413224)*/,
    5,-13, 6,10/*mean (0.109256), correlation (0.408646)*/,
    2,-12, 2,3/*mean (0.131691), correlation (0.416076)*/,
    3,8, 4,-6/*mean (0.165081), correlation (0.417569)*/,
    2,6, 12,-13/*mean (0.171874), correlation (0.408471)*/,
    9,-12, 10,3/*mean (0.175146), correlation (0.41296)*/,
    -8,4, -7,9/*mean (0.183682), correlation (0.402956)*/,
    -11,12, -4,-6/*mean (0.184672), correlation (0.416125)*/,
    1,12, 2,-8/*mean (0.191487), correlation (0.386696)*/,
    6,-9, 7,-4/*mean (0.192668), correlation (0.394771)*/,
    2,3, 3,-2/*mean (0.200157), correlation (0.408303)*/,
    6,3, 11,0/*mean (0.204588), correlation (0.411762)*/,
    3,-3, 8,-8/*mean (0.205904), correlation (0.416294)*/,
    7,8, 9,3/*mean (0.213237), correlation (0.409306)*/,
    -11,-5, -6,-4/*mean (0.243444), correlation (0.395069)*/,
    -10,11, -5,10/*mean (0.247672), correlation (0.413392)*/,
    -5,-8, -3,12/*mean (0.24774), correlation (0.411416)*/,
    -10,5, -9,0/*mean (0.00213675), correlation (0.454003)*/,
    8,-1, 12,-6/*mean (0.0293635), correlation (0.455368)*/,
    4,-6, 6,-11/*mean (0.0404971), correlation (0.457393)*/,
    -10,12, -8,7/*mean (0.0481107), correlation (0.448364)*/,
    4,-2, 6,7/*mean (0.050641), correlation (0.455019)*/,
    -2,0, -2,12/*mean (0.0525978), correlation (0.44338)*/,
    -5,-8, -5,2/*mean (0.0629667), correlation (0.457096)*/,
    7,-6, 10,12/*mean (0.0653846), correlation (0.445623)*/,
    -9,-13, -8,-8/*mean (0.0858749), correlation (0.449789)*/,
    -5,-13, -5,-2/*mean (0.122402), correlation (0.450201)*/,
    8,-8, 9,-13/*mean (0.125416), correlation (0.453224)*/,
    -9,-11, -9,0/*mean (0.130128), correlation (0.458724)*/,
    1,-8, 1,-2/*mean (0.132467), correlation (0.440133)*/,
    7,-4, 9,1/*mean (0.132692), correlation (0.454)*/,
    -2,1, -1,-4/*mean (0.135695), correlation (0.455739)*/,
    11,-6, 12,-11/*mean (0.142904), correlation (0.446114)*/,
    -12,-9, -6,4/*mean (0.146165), correlation (0.451473)*/,
    3,7, 7,12/*mean (0.147627), correlation (0.456643)*/,
    5,5, 10,8/*mean (0.152901), correlation (0.455036)*/,
    0,-4, 2,8/*mean (0.167083), correlation (0.459315)*/,
    -9,12, -5,-13/*mean (0.173234), correlation (0.454706)*/,
    0,7, 2,12/*mean (0.18312), correlation (0.433855)*/,
    -1,2, 1,7/*mean (0.185504), correlation (0.443838)*/,
    5,11, 7,-9/*mean (0.185706), correlation (0.451123)*/,
    3,5, 6,-8/*mean (0.188968), correlation (0.455808)*/,
    -13,-4, -8,9/*mean (0.191667), correlation (0.459128)*/,
    -5,9, -3,-3/*mean (0.193196), correlation (0.458364)*/,
    -4,-7, -3,-12/*mean (0.196536), correlation (0.455782)*/,
    6,5, 8,0/*mean (0.1972), correlation (0.450481)*/,
    -7,6, -6,12/*mean (0.199438), correlation (0.458156)*/,
    -13,6, -5,-2/*mean (0.211224), correlation (0.449548)*/,
    1,-10, 3,10/*mean (0.211718), correlation (0.440606)*/,
    4,1, 8,-4/*mean (0.213034), correlation (0.443177)*/,
    -2,-2, 2,-13/*mean (0.234334), correlation (0.455304)*/,
    2,-12, 12,12/*mean (0.235684), correlation (0.443436)*/,
    -2,-13, 0,-6/*mean (0.237674), correlation (0.452525)*/,
    4,1, 9,3/*mean (0.23962), correlation (0.444824)*/,
    -6,-10, -3,-5/*mean (0.248459), correlation (0.439621)*/,
    -3,-13, -1,1/*mean (0.249505), correlation (0.456666)*/,
    7,5, 12,-11/*mean (0.00119208), correlation (0.495466)*/,
    4,-2, 5,-7/*mean (0.00372245), correlation (0.484214)*/,
    -13,9, -9,-5/*mean (0.00741116), correlation (0.499854)*/,
    7,1, 8,6/*mean (0.0208952), correlation (0.499773)*/,
    7,-8, 7,6/*mean (0.0220085), correlation (0.501609)*/,
    -7,-4, -7,1/*mean (0.0233806), correlation (0.496568)*/,
    -8,11, -7,-8/*mean (0.0236505), correlation (0.489719)*/,
    -13,6, -12,-8/*mean (0.0268781), correlation (0.503487)*/,
    2,4, 3,9/*mean (0.0323324), correlation (0.501938)*/,
    10,-5, 12,3/*mean (0.0399235), correlation (0.494029)*/,
    -6,-5, -6,7/*mean (0.0420153), correlation (0.486579)*/,
    8,-3, 9,-8/*mean (0.0548021), correlation (0.484237)*/,
    2,-12, 2,8/*mean (0.0616622), correlation (0.496642)*/,
    -11,-2, -10,3/*mean (0.0627755), correlation (0.498563)*/,
    -12,-13, -7,-9/*mean (0.0829622), correlation (0.495491)*/,
    -11,0, -10,-5/*mean (0.0843342), correlation (0.487146)*/,
    5,-3, 11,8/*mean (0.0929937), correlation (0.502315)*/,
    -2,-13, -1,12/*mean (0.113327), correlation (0.48941)*/,
    -1,-8, 0,9/*mean (0.132119), correlation (0.467268)*/,
    -13,-11, -12,-5/*mean (0.136269), correlation (0.498771)*/,
    -10,-2, -10,11/*mean (0.142173), correlation (0.498714)*/,
    -3,9, -2,-13/*mean (0.144141), correlation (0.491973)*/,
    2,-3, 3,2/*mean (0.14892), correlation (0.500782)*/,
    -9,-13, -4,0/*mean (0.150371), correlation (0.498211)*/,
    -4,6, -3,-10/*mean (0.152159), correlation (0.495547)*/,
    -4,12, -2,-7/*mean (0.156152), correlation (0.496925)*/,
    -6,-11, -4,9/*mean (0.15749), correlation (0.499222)*/,
    6,-3, 6,11/*mean (0.159211), correlation (0.503821)*/,
    -13,11, -5,5/*mean (0.162427), correlation (0.501907)*/,
    11,11, 12,6/*mean (0.16652), correlation (0.497632)*/,
    7,-5, 12,-2/*mean (0.169141), correlation (0.484474)*/,
    -1,12, 0,7/*mean (0.169456), correlation (0.495339)*/,
    -4,-8, -3,-2/*mean (0.171457), correlation (0.487251)*/,
    -7,1, -6,7/*mean (0.175), correlation (0.500024)*/,
    -13,-12, -8,-13/*mean (0.175866), correlation (0.497523)*/,
    -7,-2, -6,-8/*mean (0.178273), correlation (0.501854)*/,
    -8,5, -6,-9/*mean (0.181107), correlation (0.494888)*/,
    -5,-1, -4,5/*mean (0.190227), correlation (0.482557)*/,
    -13,7, -8,10/*mean (0.196739), correlation (0.496503)*/,
    1,5, 5,-13/*mean (0.19973), correlation (0.499759)*/,
    1,0, 10,-13/*mean (0.204465), correlation (0.49873)*/,
    9,12, 10,-1/*mean (0.209334), correlation (0.49063)*/,
    5,-8, 10,-9/*mean (0.211134), correlation (0.503011)*/,
    -1,11, 1,-13/*mean (0.212), correlation (0.499414)*/,
    -9,-3, -6,2/*mean (0.212168), correlation (0.480739)*/,
    -1,-10, 1,12/*mean (0.212731), correlation (0.502523)*/,
    -13,1, -8,-10/*mean (0.21327), correlation (0.489786)*/,
    8,-11, 10,-6/*mean (0.214159), correlation (0.488246)*/,
    2,-13, 3,-6/*mean (0.216993), correlation (0.50287)*/,
    7,-13, 12,-9/*mean (0.223639), correlation (0.470502)*/,
    -10,-10, -5,-7/*mean (0.224089), correlation (0.500852)*/,
    -10,-8, -8,-13/*mean (0.228666), correlation (0.502629)*/,
    4,-6, 8,5/*mean (0.22906), correlation (0.498305)*/,
    3,12, 8,-13/*mean (0.233378), correlation (0.503825)*/,
    -4,2, -3,-3/*mean (0.234323), correlation (0.476692)*/,
    5,-13, 10,-12/*mean (0.236392), correlation (0.475462)*/,
    4,-13, 5,-1/*mean (0.236842), correlation (0.504132)*/,
    -9,9, -4,3/*mean (0.236977), correlation (0.497739)*/,
    0,3, 3,-9/*mean (0.24314), correlation (0.499398)*/,
    -12,1, -6,1/*mean (0.243297), correlation (0.489447)*/,
    3,2, 4,-8/*mean (0.00155196), correlation (0.553496)*/,
    -10,-10, -10,9/*mean (0.00239541), correlation (0.54297)*/,
    8,-13, 12,12/*mean (0.0034413), correlation (0.544361)*/,
    -8,-12, -6,-5/*mean (0.003565), correlation (0.551225)*/,
    2,2, 3,7/*mean (0.00835583), correlation (0.55285)*/,
    10,6, 11,-8/*mean (0.00885065), correlation (0.540913)*/,
    6,8, 8,-12/*mean (0.0101552), correlation (0.551085)*/,
    -7,10, -6,5/*mean (0.0102227), correlation (0.533635)*/,
    -3,-9, -3,9/*mean (0.0110211), correlation (0.543121)*/,
    -1,-13, -1,5/*mean (0.0113473), correlation (0.550173)*/,
    -3,-7, -3,4/*mean (0.0140913), correlation (0.554774)*/,
    -8,-2, -8,3/*mean (0.017049), correlation (0.55461)*/,
    4,2, 12,12/*mean (0.01778), correlation (0.546921)*/,
    2,-5, 3,11/*mean (0.0224022), correlation (0.549667)*/,
    6,-9, 11,-13/*mean (0.029161), correlation (0.546295)*/,
    3,-1, 7,12/*mean (0.0303081), correlation (0.548599)*/,
    11,-1, 12,4/*mean (0.0355151), correlation (0.523943)*/,
    -3,0, -3,6/*mean (0.0417904), correlation (0.543395)*/,
    4,-11, 4,12/*mean (0.0487292), correlation (0.542818)*/,
    2,-4, 2,1/*mean (0.0575124), correlation (0.554888)*/,
    -10,-6, -8,1/*mean (0.0594242), correlation (0.544026)*/,
    -13,7, -11,1/*mean (0.0597391), correlation (0.550524)*/,
    -13,12, -11,-13/*mean (0.0608974), correlation (0.55383)*/,
    6,0, 11,-13/*mean (0.065126), correlation (0.552006)*/,
    0,-1, 1,4/*mean (0.074224), correlation (0.546372)*/,
    -13,3, -9,-2/*mean (0.0808592), correlation (0.554875)*/,
    -9,8, -6,-3/*mean (0.0883378), correlation (0.551178)*/,
    -13,-6, -8,-2/*mean (0.0901035), correlation (0.548446)*/,
    5,-9, 8,10/*mean (0.0949843), correlation (0.554694)*/,
    2,7, 3,-9/*mean (0.0994152), correlation (0.550979)*/,
    -1,-6, -1,-1/*mean (0.10045), correlation (0.552714)*/,
    9,5, 11,-2/*mean (0.100686), correlation (0.552594)*/,
    11,-3, 12,-8/*mean (0.101091), correlation (0.532394)*/,
    3,0, 3,5/*mean (0.101147), correlation (0.525576)*/,
    -1,4, 0,10/*mean (0.105263), correlation (0.531498)*/,
    3,-6, 4,5/*mean (0.110785), correlation (0.540491)*/,
    -13,0, -10,5/*mean (0.112798), correlation (0.536582)*/,
    5,8, 12,11/*mean (0.114181), correlation (0.555793)*/,
    8,9, 9,-6/*mean (0.117431), correlation (0.553763)*/,
    7,-4, 8,-12/*mean (0.118522), correlation (0.553452)*/,
    -10,4, -10,9/*mean (0.12094), correlation (0.554785)*/,
    7,3, 12,4/*mean (0.122582), correlation (0.555825)*/,
    9,-7, 10,-2/*mean (0.124978), correlation (0.549846)*/,
    7,0, 12,-2/*mean (0.127002), correlation (0.537452)*/,
    -1,-6, 0,-11/*mean (0.127148), correlation (0.547401)*/
};


static void makeRandomPattern(int patchSize, Point* pattern, int npoints)
{
    RNG rng(0x34985739); // we always start with a fixed seed,
                         // to make patterns the same on each run
    for( int i = 0; i < npoints; i++ )
    {
        pattern[i].x = rng.uniform(-patchSize/2, patchSize/2+1);
        pattern[i].y = rng.uniform(-patchSize/2, patchSize/2+1);
    }
}


static inline float getScale(int level, int firstLevel, double scaleFactor)
{
    return (float)std::pow(scaleFactor, (double)(level - firstLevel));
}


class ORB_Impl : public ORB
{
public:
    //ORB_Impl构造函数
    explicit ORB_Impl(int _nfeatures, float _scaleFactor, int _nlevels, int _edgeThreshold,
             int _firstLevel, int _WTA_K, int _scoreType, int _patchSize, int _fastThreshold) :
        nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
        edgeThreshold(_edgeThreshold), firstLevel(_firstLevel), wta_k(_WTA_K),
        scoreType(_scoreType), patchSize(_patchSize), fastThreshold(_fastThreshold)
    {}

    void setMaxFeatures(int maxFeatures) { nfeatures = maxFeatures; }
    int getMaxFeatures() const { return nfeatures; }

    void setScaleFactor(double scaleFactor_) { scaleFactor = scaleFactor_; }
    double getScaleFactor() const { return scaleFactor; }

    void setNLevels(int nlevels_) { nlevels = nlevels_; }
    int getNLevels() const { return nlevels; }

    void setEdgeThreshold(int edgeThreshold_) { edgeThreshold = edgeThreshold_; }
    int getEdgeThreshold() const { return edgeThreshold; }

    void setFirstLevel(int firstLevel_) { firstLevel = firstLevel_; }
    int getFirstLevel() const { return firstLevel; }

    void setWTA_K(int wta_k_) { wta_k = wta_k_; }
    int getWTA_K() const { return wta_k; }

    void setScoreType(int scoreType_) { scoreType = scoreType_; }
    int getScoreType() const { return scoreType; }

    void setPatchSize(int patchSize_) { patchSize = patchSize_; }
    int getPatchSize() const { return patchSize; }

    void setFastThreshold(int fastThreshold_) { fastThreshold = fastThreshold_; }
    int getFastThreshold() const { return fastThreshold; }

    // returns the descriptor size in bytes
    int descriptorSize() const;
    // returns the descriptor type
    int descriptorType() const;
    // returns the default norm type
    int defaultNorm() const;

    // Compute the ORB_Impl features and descriptors on an image
    void detectAndCompute( InputArray image, InputArray mask, std::vector<KeyPoint>& keypoints,
                     OutputArray descriptors, bool useProvidedKeypoints=false );

protected:

    int nfeatures;//表示最多保留多少个特征点默认值500
    double scaleFactor;//图像缩放的尺度，默认值是1.2
    int nlevels;//将图像分为多少层，默认值是8
    int edgeThreshold;//距离图像边缘多少不进入到检测的范围，默认值是31，必须与patchSize相同
    int firstLevel;//从金字塔第几层开始检测特征点，默认值是0
    int wta_k;//在使用brief算法计算描述子需要随机采样进行比较，这里表示随机采样进行比较的像素个数，默认值是2
    int scoreType;//用哪种特征点检测算法来评判分数，默认值是ORB::HARRIS_SCORE
    int patchSize;//计算描述子是在哪个范围内计算
    int fastThreshold;//默认值是20
};

int ORB_Impl::descriptorSize() const
{
    return kBytes;
}

int ORB_Impl::descriptorType() const
{
    return CV_8U;
}

int ORB_Impl::defaultNorm() const
{
    return NORM_HAMMING;
}

#ifdef HAVE_OPENCL
static void uploadORBKeypoints(const std::vector<KeyPoint>& src, std::vector<Vec3i>& buf, OutputArray dst)
{
    size_t i, n = src.size();
    buf.resize(std::max(buf.size(), n));
    for( i = 0; i < n; i++ )
        buf[i] = Vec3i(cvRound(src[i].pt.x), cvRound(src[i].pt.y), src[i].octave);
    copyVectorToUMat(buf, dst);
}

typedef union if32_t
{
    int i;
    float f;
}
if32_t;

static void uploadORBKeypoints(const std::vector<KeyPoint>& src,
                               const std::vector<float>& layerScale,
                               std::vector<Vec4i>& buf, OutputArray dst)
{
    size_t i, n = src.size();
    buf.resize(std::max(buf.size(), n));
    for( i = 0; i < n; i++ )
    {
        int z = src[i].octave;
        float scale = 1.f/layerScale[z];
        if32_t angle;
        angle.f = src[i].angle;
        buf[i] = Vec4i(cvRound(src[i].pt.x*scale), cvRound(src[i].pt.y*scale), z, angle.i);
    }
    copyVectorToUMat(buf, dst);
}
#endif

/** Compute the ORB_Impl keypoints on an image
 * @param image_pyramid the image pyramid to compute the features and descriptors on
 * @param mask_pyramid the masks to apply at every level
 * @param keypoints the resulting keypoints, clustered per level
 */
 //computeKeyPoints函数
 //imagePyramid输入的金字塔各层图像，
 //默认情况下uimagePyramid和maskPyramid=空
 //layerInfo:整个数据结构中的这个矩阵，不包含边界的尺寸
 //ulayerInfo默认情况下为空
 //layerScale:序号是金字塔的层数，内容是缩放的比例即:1，1.2，1.2的平方。。。。1.2的7次方
 //allKeypoints输出的特征点
 //nfeatures:最多保留的特征点个数 默认值是500
 //scaleFactor图像的缩放比例，默认请情况下是1.2
 //edgeThreshold=距离图像边缘多少不进入到检测的范围，默认值是31，必须与patchSize相同
 //patchSize默认情况下是31
 //scoreType默认情况下是ORB::HARRIS_SCORE
 //useOCL一般不使用cuda时都为false
 //fastThreshold默认值是20
static void computeKeyPoints(const Mat& imagePyramid,
                             const UMat& uimagePyramid,
                             const Mat& maskPyramid,
                             const std::vector<Rect>& layerInfo,
                             const UMat& ulayerInfo,
                             const std::vector<float>& layerScale,
                             std::vector<KeyPoint>& allKeypoints,
                             int nfeatures, double scaleFactor,
                             int edgeThreshold, int patchSize, int scoreType,
                             bool useOCL, int fastThreshold  )
{
#ifndef HAVE_OPENCL
    (void)uimagePyramid;(void)ulayerInfo;(void)useOCL;
#endif

    int i, nkeypoints, level, nlevels = (int)layerInfo.size();
    std::vector<int> nfeaturesPerLevel(nlevels);//这个参数存储的是每层金字塔最多保留的特征点数目

    // fill the extractors and descriptors for the corresponding scales
    float factor = (float)(1.0 / scaleFactor);
    float ndesiredFeaturesPerScale = nfeatures*(1 - factor)/(1 - (float)std::pow((double)factor, (double)nlevels));//这个参数存储的是每层金字塔最多保留的特征点

    int sumFeatures = 0;
    for( level = 0; level < nlevels-1; level++ )
    {
        nfeaturesPerLevel[level] = cvRound(ndesiredFeaturesPerScale);
        sumFeatures += nfeaturesPerLevel[level];
        ndesiredFeaturesPerScale *= factor;
    }
    nfeaturesPerLevel[nlevels-1] = std::max(nfeatures - sumFeatures, 0);

    // Make sure we forget about what is too close to the boundary
    //edge_threshold_ = std::max(edge_threshold_, patch_size_/2 + kKernelWidth / 2 + 2);

    // pre-compute the end of a row in a circular patch
    //下面的目的是求umax---------------------这个参数会在ICAngles函数中用到
    int halfPatchSize = patchSize / 2;//15
    std::vector<int> umax(halfPatchSize + 2);//17

    int v, v0, vmax = cvFloor(halfPatchSize * std::sqrt(2.f) / 2 + 1);//11
    int vmin = cvCeil(halfPatchSize * std::sqrt(2.f) / 2);//11
    for (v = 0; v <= vmax; ++v)
        umax[v] = cvRound(std::sqrt((double)halfPatchSize * halfPatchSize - v * v));//15,15,15,15,14,14,14,13,13,12,11,10

    // Make sure we are symmetric,
    for (v = halfPatchSize, v0 = 0; v >= vmin; --v)
    {
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }
    //经过上述函数后umax=15，15，15，15，14，14，14，13，13，12，11，10，9，8，6，3，0
    //----------------------------------
    allKeypoints.clear();//存储所有层金字塔的特征点
    std::vector<KeyPoint> keypoints;//存储某曾金字塔的特征点
    std::vector<int> counters(nlevels);//这个参数记录每层金字塔图像中检测到的特征点数量
    keypoints.reserve(nfeaturesPerLevel[0]*2);

    for( level = 0; level < nlevels; level++ )
    {
        int featuresNum = nfeaturesPerLevel[level];//提取出这层金字塔最多检测的特征点数目
        Mat img = imagePyramid(layerInfo[level]);//提取出不带边界的这层的图像
        Mat mask = maskPyramid.empty() ? Mat() : maskPyramid(layerInfo[level]);//构造一个和这层图像一样大的mask

        // Detect FAST features, 20 is a good threshold
        {
        //第一个参数表示中心像素的像素值和该像素周围的像素值之差的阈值
        //第二个参数表示使用非最大值抑制，
        Ptr<FastFeatureDetector> fd = FastFeatureDetector::create(fastThreshold, true);
        //提取当前层图像的特征点，此处的mask应该是我们需要提取的图像的哪部分的区域，此处我们提取的是整个图像的特征点
        fd->detect(img, keypoints, mask);
        }

        // Remove keypoints very close to the border
        //剔除那些距离图像边缘31个像素的特征点
        KeyPointsFilter::runByImageBorder(keypoints, img.size(), edgeThreshold);

        // Keep more points than necessary as FAST does not give amazing corners
        //scoreType默认是HARRIS_SCORE
        KeyPointsFilter::retainBest(keypoints, scoreType == ORB_Impl::HARRIS_SCORE ? 2 * featuresNum : featuresNum);

        nkeypoints = (int)keypoints.size();
        counters[level] = nkeypoints;

        float sf = layerScale[level];//得到的是1，1.2，1.2的平方。。。。1.2的7次方
        for( i = 0; i < nkeypoints; i++ )
        {
            keypoints[i].octave = level;//更新这个特征点是在哪一层金字塔被检测出来的
            keypoints[i].size = patchSize*sf;//
        }

        std::copy(keypoints.begin(), keypoints.end(), std::back_inserter(allKeypoints));//将这一层检测出来的特征点保存在allKeypoints变量中
    }

    std::vector<Vec3i> ukeypoints_buf;

    nkeypoints = (int)allKeypoints.size();
    if(nkeypoints == 0)
    {
        return;
    }
    Mat responses;
    UMat ukeypoints, uresponses(1, nkeypoints, CV_32F);

    // Select best features using the Harris cornerness (better scoring than FAST)
    if( scoreType == ORB_Impl::HARRIS_SCORE )//默认情况下进入这个选项
    {
    /*这里我们把使用cuda进行编码的程序注释掉 方便看程序
#ifdef HAVE_OPENCL
        if( useOCL )
        {
            uploadORBKeypoints(allKeypoints, ukeypoints_buf, ukeypoints);
            useOCL = ocl_HarrisResponses( uimagePyramid, ulayerInfo, ukeypoints,
                                          uresponses, nkeypoints, 7, HARRIS_K );
            if( useOCL )
            {
                CV_IMPL_ADD(CV_IMPL_OCL);
                uresponses.copyTo(responses);
                for( i = 0; i < nkeypoints; i++ )
                    allKeypoints[i].response = responses.at<float>(i);
            }
        }

        if( !useOCL )
#endif
*/      
         HarrisResponses(imagePyramid, layerInfo, allKeypoints, 7, HARRIS_K);//此函数主要是更新了allKeypoints中的response响应，评价fast角点的特征点程度

        std::vector<KeyPoint> newAllKeypoints;
        newAllKeypoints.reserve(nfeaturesPerLevel[0]*nlevels);

        int offset = 0;
	//根据我们之前得到的新的harris评测指标再来筛选fast角点,从而更新allKeypoints中的特征点
        for( level = 0; level < nlevels; level++ )
        {
            int featuresNum = nfeaturesPerLevel[level];
            nkeypoints = counters[level];
            keypoints.resize(nkeypoints);
            std::copy(allKeypoints.begin() + offset,  allKeypoints.begin() + offset + nkeypoints,  keypoints.begin());
            offset += nkeypoints;

            //cull to the final desired level, using the new Harris scores.
            //根据我们之前得到的新的harris评测指标再来筛选fast角点
            KeyPointsFilter::retainBest(keypoints, featuresNum);

            std::copy(keypoints.begin(), keypoints.end(), std::back_inserter(newAllKeypoints));
        }
        std::swap(allKeypoints, newAllKeypoints);
    }

    nkeypoints = (int)allKeypoints.size();
/*这里我们把使用cuda进行编码的程序注释掉 方便看程序
#ifdef HAVE_OPENCL
    if( useOCL )
    {
        UMat uumax;
        if( useOCL )
            copyVectorToUMat(umax, uumax);

        uploadORBKeypoints(allKeypoints, ukeypoints_buf, ukeypoints);
        useOCL = ocl_ICAngles(uimagePyramid, ulayerInfo, ukeypoints, uresponses, uumax,
                              nkeypoints, halfPatchSize);

        if( useOCL )
        {
            CV_IMPL_ADD(CV_IMPL_OCL);
            uresponses.copyTo(responses);
            for( i = 0; i < nkeypoints; i++ )
                allKeypoints[i].angle = responses.at<float>(i);
        }
    }

    if( !useOCL )
#endif
*/
    {
       //搜索"ICAngles函数"
       //主要是更新allKeypoints中特征点的方向
        ICAngles(imagePyramid, layerInfo, allKeypoints, umax, halfPatchSize);
    }

    for( i = 0; i < nkeypoints; i++ )
    {
        float scale = layerScale[allKeypoints[i].octave];
        allKeypoints[i].pt *= scale;//根据特征点被检测到的金字塔层数，将在某一层的金字塔图像乘以比例还原到原图像中
    }
}


/** Compute the ORB_Impl features and descriptors on an image
 * @param img the image to compute the features and descriptors on
 * @param mask the mask to apply
 * @param keypoints the resulting keypoints
 * @param descriptors the resulting descriptors
 * @param do_keypoints if true, the keypoints are computed, otherwise used as an input
 * @param do_descriptors if true, also computes the descriptors
 */
 //搜索"detectAndCompute头文件"
 //useProvidedKeypoints默认值是false
void ORB_Impl::detectAndCompute( InputArray _image, InputArray _mask,
                                 					   std::vector<KeyPoint>& keypoints,
                               					  OutputArray _descriptors, bool useProvidedKeypoints )
{
    CV_INSTRUMENT_REGION()//OpenCV相关算法表现性能测试框架,默认不应用次选项，所以不消耗资源。

    CV_Assert(patchSize >= 2);

    bool do_keypoints = !useProvidedKeypoints;
    bool do_descriptors = _descriptors.needed();

    if( (!do_keypoints && !do_descriptors) || _image.empty() )
        return;

    //ROI handling
    const int HARRIS_BLOCK_SIZE = 9;
    int halfPatchSize = patchSize / 2;//patchSize=31，halfPatchSize=15
    // sqrt(2.0) is for handling patch rotation
    int descPatchSize = cvCeil(halfPatchSize*sqrt(2.0));//15*根号2 22
    int border = std::max(edgeThreshold, std::max(descPatchSize, HARRIS_BLOCK_SIZE/2))+1;//max(31,max(22，4))=31

    bool useOCL = ocl::useOpenCL() && OCL_FORCE_CHECK(_image.isUMat() || _descriptors.isUMat());//OCL是一个用于GPU执行的库，一般我们用不到

    Mat image = _image.getMat(), mask = _mask.getMat();
    if( image.type() != CV_8UC1 )
        cvtColor(_image, image, COLOR_BGR2GRAY);

    int i, level, nLevels = this->nlevels, nkeypoints = (int)keypoints.size();
    bool sortedByLevel = true;

    if( !do_keypoints )//默认参数的情况下不进入这个条件
    {
        // if we have pre-computed keypoints, they may use more levels than it is set in parameters
        // !!!TODO!!! implement more correct method, independent from the used keypoint detector.
        // Namely, the detector should provide correct size of each keypoint. Based on the keypoint size
        // and the algorithm used (i.e. BRIEF, running on 31x31 patches) we should compute the approximate
        // scale-factor that we need to apply. Then we should cluster all the computed scale-factors and
        // for each cluster compute the corresponding image.
        //
        // In short, ultimately the descriptor should
        // ignore octave parameter and deal only with the keypoint size.
        nLevels = 0;
        for( i = 0; i < nkeypoints; i++ )
        {
            level = keypoints[i].octave;
            CV_Assert(level >= 0);
            if( i > 0 && level < keypoints[i-1].octave )
                sortedByLevel = false;
            nLevels = std::max(nLevels, level);
        }
        nLevels++;
    }

    std::vector<Rect> layerInfo(nLevels); //存储每层的矩形
    std::vector<int> layerOfs(nLevels);
    std::vector<float> layerScale(nLevels); //序号是金字塔的层数，内容是缩放的比例即:1，1.2，1.2的平方。。。。1.2的7次方
    Mat imagePyramid, maskPyramid;
    UMat uimagePyramid, ulayerInfo;

    int level_dy = image.rows + border*2;//480+31*2=542
    Point level_ofs(0,0);
    //这个参数中存储的是整个金字塔图像的尺寸，详见算法实现文档
    Size bufSize((image.cols + border*2 + 15) & -16, 0);//&-16的目的是让低四位置零，第一个参数是宽=816 第二个参数是高,我感觉目的是得到一个最小的数值使其能够被16除尽头 
    //这里我们默认图像的col=752,row=480
    for( level = 0; level < nLevels; level++ )
    {
        float scale = getScale(level, firstLevel, scaleFactor);//对于默认设置则其中存储的是1，1.2，1.2的平方。。。。1.2的7次方
        layerScale[level] = scale;
        Size sz(cvRound(image.cols/scale), cvRound(image.rows/scale));//不同尺度下的图像尺寸
        Size wholeSize(sz.width + border*2, sz.height + border*2);//(814,542)
        if( level_ofs.x + wholeSize.width > bufSize.width )
        {
            level_ofs = Point(0, level_ofs.y + level_dy);
            level_dy = wholeSize.height;
        }

        Rect linfo(level_ofs.x + border, level_ofs.y + border, sz.width, sz.height);//前两个参数表示这个方向的左上角的坐标
        layerInfo[level] = linfo;//需要更新的!!!!!!!!!!存储的是在整个数据结构中的这个矩阵，不包含边界的尺寸
        layerOfs[level] = linfo.y*bufSize.width + linfo.x;//需要更新的!!!!!!!!!!存储的是这个尺度图像下的起始地址
        level_ofs.x += wholeSize.width;
    }
    bufSize.height = level_ofs.y + level_dy;

    imagePyramid.create(bufSize, CV_8U);//这个参数中存储了所有层金字塔的图像
    if( !mask.empty() )
        maskPyramid.create(bufSize, CV_8U);

    Mat prevImg = image, prevMask = mask;

    // Pre-compute the scale pyramids
    //遍历所有的金字塔层的图像
    for (level = 0; level < nLevels; ++level)
    {
        Rect linfo = layerInfo[level];
        Size sz(linfo.width, linfo.height);//不包含边界的尺寸
        Size wholeSize(sz.width + border*2, sz.height + border*2);
        Rect wholeLinfo = Rect(linfo.x - border, linfo.y - border, wholeSize.width, wholeSize.height);
        Mat extImg = imagePyramid(wholeLinfo), extMask;//在图像金字塔中获取包含边界的图像extImg
        Mat currImg = extImg(Rect(border, border, sz.width, sz.height)), currMask;//从包含边界的extImg图像中提取出不包含边界的currImg

        if( !mask.empty() )
        {
            extMask = maskPyramid(wholeLinfo);
            currMask = extMask(Rect(border, border, sz.width, sz.height));
        }

        // Compute the resized image
        if( level != firstLevel )//不是第一层的图像
        {
            resize(prevImg, currImg, sz, 0, 0, INTER_LINEAR);//这里对图像进行缩放
            if( !mask.empty() )
            {
                resize(prevMask, currMask, sz, 0, 0, INTER_LINEAR);
                if( level > firstLevel )
                    threshold(currMask, currMask, 254, 0, THRESH_TOZERO);
            }
            //对缩放得到的图像进行边界填充
            copyMakeBorder(currImg, extImg, border, border, border, border,BORDER_REFLECT_101+BORDER_ISOLATED);
            if (!mask.empty())
                copyMakeBorder(currMask, extMask, border, border, border, border,BORDER_CONSTANT+BORDER_ISOLATED);
        }
        else//这是第一层的图像
        { 
      	     //扩充src的边缘，将图像变大，然后以各种外插方式自动填充图像边界
            //image=输入的图像，extImg=输出的图像
            //border表示上下左右需要扩充多大的边缘
            copyMakeBorder(image, extImg, border, border, border, border,BORDER_REFLECT_101);
            if( !mask.empty() )//不进入这个条件
                copyMakeBorder(mask, extMask, border, border, border, border,BORDER_CONSTANT+BORDER_ISOLATED);
        }
        prevImg = currImg;
        prevMask = currMask;
    }

    if( useOCL )//一般不进入这个条件
        copyVectorToUMat(layerOfs, ulayerInfo);

    if( do_keypoints )//进入这个条件 开始检测特征点
    {
        if( useOCL )//一般不进入这个条件
            imagePyramid.copyTo(uimagePyramid);
         //double t, tf = getTickFrequency();
         //t = (double)getTickCount();
        // Get keypoints, those will be far enough from the border that no check will be required for the descriptor
        //搜索"computeKeyPoints函数"!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        //主要输出的是keypoints，得到了特征点在原图像的像素坐标，响应程度，方向和size
        computeKeyPoints(imagePyramid, uimagePyramid, maskPyramid, layerInfo, ulayerInfo, layerScale, keypoints,
                                  nfeatures, scaleFactor, edgeThreshold, patchSize, scoreType, useOCL, fastThreshold);
		 
        //t = (double)getTickCount() - t;
	 //printf("pyramid construction time: %g\n", t*1000./tf);
    }
    else
    {
        KeyPointsFilter::runByImageBorder(keypoints, image.size(), edgeThreshold);

        if( !sortedByLevel )
        {
            std::vector<std::vector<KeyPoint> > allKeypoints(nLevels);
            nkeypoints = (int)keypoints.size();
            for( i = 0; i < nkeypoints; i++ )
            {
                level = keypoints[i].octave;
                CV_Assert(0 <= level);
                allKeypoints[level].push_back(keypoints[i]);
            }
            keypoints.clear();
            for( level = 0; level < nLevels; level++ )
                std::copy(allKeypoints[level].begin(), allKeypoints[level].end(), std::back_inserter(keypoints));
        }
    }

    if( do_descriptors )//进入这个条件计算描述子
    {
        int dsize = descriptorSize();//设置的是32，描述子一共32维度

        nkeypoints = (int)keypoints.size();
        if( nkeypoints == 0 )
        {
            _descriptors.release();
            return;
        }

        _descriptors.create(nkeypoints, dsize, CV_8U);
        std::vector<Point> pattern;

        const int npoints = 512;
        Point patternbuf[npoints];
        const Point* pattern0 = (const Point*)bit_pattern_31_;

        if( patchSize != 31 )
        {
            pattern0 = patternbuf;
            makeRandomPattern(patchSize, patternbuf, npoints);
        }

        CV_Assert( wta_k == 2 || wta_k == 3 || wta_k == 4 );

        if( wta_k == 2 )//默认设置就是2，brief进行描述子计算时就是两个点进行比较
            std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));//这里将坐标的存储形式保存为点的存储形式，1个点对应两个坐标
        else
        {
            int ntuples = descriptorSize()*4;
            initializeOrbPattern(pattern0, pattern, ntuples, wta_k, npoints);
        }

        for( level = 0; level < nLevels; level++ )
        {
            // preprocess the resized image
            Mat workingMat = imagePyramid(layerInfo[level]);//提取出金字塔对应的某层图像

            //boxFilter(working_mat, working_mat, working_mat.depth(), Size(5,5), Point(-1,-1), true, BORDER_REFLECT_101);
            //对workingMat图像进行模糊化处理。Size(7, 7)是模糊核的大小。2,2是x和y方向的方差，BORDER_REFLECT_101是插值方法。
            GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);
        }
/*为了阅读代码方便 我们将有关cudad执行的程序都注释掉 
#ifdef HAVE_OPENCL//opencl是一个并行计算的库应该是不进入这个条件
        if( useOCL )
        {
            imagePyramid.copyTo(uimagePyramid);
            std::vector<Vec4i> kptbuf;
            UMat ukeypoints, upattern;
            copyVectorToUMat(pattern, upattern);
            uploadORBKeypoints(keypoints, layerScale, kptbuf, ukeypoints);

            UMat udescriptors = _descriptors.getUMat();
            useOCL = ocl_computeOrbDescriptors(uimagePyramid, ulayerInfo,
                                               ukeypoints, udescriptors, upattern,
                                               nkeypoints, dsize, wta_k);
            if(useOCL)
            {
                CV_IMPL_ADD(CV_IMPL_OCL);
            }
        }

        if( !useOCL )
#endif
*/
        {
            Mat descriptors = _descriptors.getMat();//这个参数是即将输出的描述子矩阵
	     //搜索"computeOrbDescriptors函数"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	     //最终输出的是descriptors
            computeOrbDescriptors(imagePyramid, layerInfo, layerScale,keypoints, descriptors, pattern, dsize, wta_k);
        }
    }
}

//搜索"orb create头文件"
//Ptr<ORB> create(int nfeatures=500, float scaleFactor=1.2f, int nlevels=8, int edgeThreshold=31,int firstLevel=0, int WTA_K=2, int scoreType=ORB::HARRIS_SCORE, int patchSize=31, int fastThreshold=20);
Ptr<ORB> ORB::create(int nfeatures, float scaleFactor, int nlevels, int edgeThreshold,
           int firstLevel, int wta_k, int scoreType, int patchSize, int fastThreshold)
{
    //这里调用了ORB_Impl的构造函数
    //搜索"ORB_Impl构造函数"
    return makePtr<ORB_Impl>(nfeatures, scaleFactor, nlevels, edgeThreshold,
                             firstLevel, wta_k, scoreType, patchSize, fastThreshold);
}

}
