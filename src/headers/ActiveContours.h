/* 
 * File:   ActiveContours.h
 * Author: olmozavala
 *
 * Created on October 10, 2011, 10:08 AM
 */

#ifndef ACTIVECONTOURS_H
#define	ACTIVECONTOURS_H

#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <chrono>

#include "FreeImage.h"
#include "CLManager/CLManager.h"
#include "FileManager/FileManager.h"
#include "ImageManager/ImageManager.h"
#include "CLManager/ErrorCodes.h"
#include "Timers/timing.h"

class ActiveContours {
public:
    ActiveContours();
    ActiveContours(const ActiveContours& orig);
    virtual ~ActiveContours();

    void loadProgram(int iter, float alpha, float dt, Timings& ts);

    void iterate(int numIterations, bool useAllBands);
    void initImagesArraysAndBuffers(GLuint& gl_text_input, GLuint& gl_text_output,
						int locwidth, int locheight, int locdepth, bool cleanBuffers);
    void runSDF(); 
    void create3DMask(int width, int height, int depth,
    int colStart, int colEnd, int rowStart, int rowEnd, int depthStart, int depthEnd);

private:
    
	void printNDRanges(int cols, int rows, int z, int grp_cols, int grp_rows, int grp_z);
	void printBuffer(cl::Buffer& buf, int size, vector<cl::Event> vecPrev);
	void printBufferArray(cl::Buffer& buf, int size, int width, int height,
            vector<cl::Event> vecPrev,int* slides, int sizeOfArray);
	void printBuffer(cl::Buffer& buf, int size, int offset, int width, int height, vector<cl::Event> vecPrev);
    int SDFoz(CLManager cl, cl::Context context, cl::CommandQueue queue, cl::Program program);
    int SDFVoro(CLManager cl, cl::Context context, cl::CommandQueue queue, cl::Program program);
    int iterateAvgInAndOut(int width, int height, int grp_size_x, int grp_size_y, int arraySize);
    void tic(Timer* timer);
    void toc(Timer* timer);
    
	void copySegToText();
    cl::Event compDphiDt(vector<cl::Event> vecEvPrev);
    cl::Event compCurvature(vector<cl::Event> vecEvPrev);
    cl::Event compF(vector<cl::Event> vecEvPrev);
    cl::Event compNewPhi(vector<cl::Event> vecEvPrev);
    cl::Event smoothPhi(vector<cl::Event> vecEvPrev, float dt);
    
	cl::Event copyBufToImg(cl::Buffer& buf, cl::Image3D& text, vector<cl::Event> vecEvPrev, bool bufHasAllBands);
	cl::Event copyTextToBuf(cl::Image3DGL& text, cl::Buffer& buf, vector<cl::Event> vecEvPrev);

	cl::Event compReduce(cl::Buffer& buf, cl::Buffer& buf_out, 
						bool absVal, vector<cl::Event> vecEvPrev);
    
	cl::Event compAvgInAndOut(cl::Buffer& buf_phi, cl::Buffer& buf_img_in,
        vector<cl::Event> vecPrevEvents);
    
    Timings* ts;
    Timer* tm_copyGlToBuffer;
    Timer* tm_avgInOut;
    Timer* tm_curvature;
    Timer* tm_F;
    Timer* tm_maxF;
    Timer* tm_DphiDt;
    Timer* tm_maxDphiDt;
    Timer* tm_phi;
    Timer* tm_smoothPhi;
    Timer* tm_copySmoothPhi;
    Timer* tm_bufToGL;


	// Here we are using width*height*depth*8
    cl::Image3DGL img_in_gl;//This is the object that holds the 3D texture of the nii image
    cl::Image3DGL img_phi_gl;//Output 
    
	std::vector<cl::Memory> cl_textures;
    
	// Here we are using width*height*depth*4
    cl::Buffer buf_avg_in_out;
    cl::Buffer buf_img_in;
    cl::Buffer buf_F;
    cl::Buffer buf_max_F;
    cl::Buffer buf_max_dphidt;
    cl::Buffer buf_dphidt;
	cl::Buffer buf_phi;
	cl::Buffer buf_smooth_phi;
	cl::Buffer buf_mask;
    cl::Buffer buf_curvature;
    
    // Used by the SDF algorithms
    vector<cl::Event> vecWriteImage;
    
    cl::size_t < 3 > origin;
    cl::size_t < 3 > region;
    
    float* arr_img_phi;
    unsigned char* arr_buf_mask;
    
    float* arr_img_out;
    float* arr_avgInOutByGrp;
    float* arr_max_F;
    float* arr_max_dphidt;
    
    //Sizes of the 3D texture
    int width;
    int height;
    int depth;
    
    // The group sizes depend on the size of the image (width and height) and 
    // on the maximum number of threads allowed on the device. 
    int grp_size_x; // Default group size on x. 
    int grp_size_y; // Default group size on y.
    int grp_size_z; // Default group size on z.
    
    // Total number of groups for each dimension
    int tot_grps_x;
    int tot_grps_y;
    int tot_grps_z;
    
    int dev_max_work_items;
	int dev_max_local_mem;
	int buf_size;//Size of the whole 3D image
    float alpha;
    float dt_smooth;
    
    cl_int err;
    cl_int res;
    
    CLManager clMan;
    //    CLProfiler prof;    
    
    cl::Context* context;
    cl::CommandQueue* queue;
    cl::Program* program;
    
    // These events will hold all the required events to compute the AVG
    vector<cl::Event> vecEvPrevAvgInOut;
    vector<cl::Event> vecEvPrevAvg;
    vector<cl::Event> vecEvPrevCurvature;
    vector<cl::Event> vecEvPrevF;
    vector<cl::Event> vecEvPrevMaxF;
    vector<cl::Event> vecEvPrevDphiDt;
    vector<cl::Event> vecEvPrevMaxDphiDt;
    vector<cl::Event> vecEvPrevNewPhi;
    vector<cl::Event> vecEvPrevSmPhi;
    vector<cl::Event> vecEvPrevSDF;
    vector<cl::Event> vecEvPrevPrinting;
    vector<cl::Event> vecEvPrevTextToBuffer;
    vector<cl::Event> vecEvPrevCopyBackTexture;
    vector<cl::Event> vecEvPrevCopyPhiBackToGL;
    vector<cl::Event> vecEvPrevCopySmoothToPhi;

    
	cl::Event evAcOGL; //Event to aquire the input texture from OpenGL
    cl::Event evImgInWrt; //Event for writing the 'in' image
    cl::Event evImgSegWrt; //Wevent for writing the 'mask' image
    cl::Event evSDF_newPhi;
    cl::Event evAvgInOut_SmoothPhi;
    cl::Event evCurvature_copySmoothToPhi;
    cl::Event evF;// When the computation of the Force has finish
    cl::Event evMaxF;
    cl::Event evDphiDt_MaxDphiDt;
	// !!!! IMPORTANT, if I use one more event I start getting seg fault CRAZY!!!
    cl::Event evCopyTextureBackGL;
    
    int currIter; //Current iteration of the algorithm
    int totalIterations;
    
    char* outputFile; //Final png file with the segmentation
};

#endif	/* ACTIVECONTOURS_H */

