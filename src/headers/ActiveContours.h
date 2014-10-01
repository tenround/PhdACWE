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

#include "FreeImage.h"
#include "CLManager/CLManager.h"
#include "FileManager/FileManager.h"
#include "ImageManager/ImageManager.h"
#include "CLManager/ErrorCodes.h"

class ActiveContours {
public:
    ActiveContours();
    ActiveContours(const ActiveContours& orig);
    virtual ~ActiveContours();

    void loadProgram(int iter, float alpha, float dt);

    void iterate(int numIterations, bool useAllBands);
    void initImagesArraysAndBuffers(GLuint& gl_text_input, GLuint& gl_text_output,
						int locwidth, int locheight, int locdepth);
    void runSDF(); 
    void create3DMask(int width, int height, int depth,
		int colStart, int colEnd, int rowStart, int rowEnd, int depthStart, int depthEnd);

private:
    
	void printNDRanges(int cols, int rows, int z, int grp_cols, int grp_rows, int grp_z);
    int SDFoz(CLManager cl, cl::Context context, cl::CommandQueue queue, cl::Program program);
    int SDFVoro(CLManager cl, cl::Context context, cl::CommandQueue queue, cl::Program program);
    int iterateAvgInAndOut(int width, int height, int grp_size_x, int grp_size_y, int arraySize);
    
    cl::Event compDphiDt(vector<cl::Event> vecEvPrev);
    cl::Event compCurvAndF(vector<cl::Event> vecEvPrev, bool useAllBands);
    cl::Event compNewPhi(vector<cl::Event> vecEvPrev);
    cl::Event smoothPhi(vector<cl::Event> vecEvPrev, float dt);
    
	cl::Event copyBufToImg(cl::Buffer& buf, cl::Image3D& text, vector<cl::Event> vecEvPrev, bool bufHasAllBands);
	cl::Event copyTextToBuf(cl::Image3DGL& text, cl::Buffer& buf, vector<cl::Event> vecEvPrev);
	void copySegToText();
    
    cl::Event compImgMaxReduction(cl::Image3D& img, vector<cl::Event> vecEvPrev,
    cl::Buffer& buf_out, int layer, bool absVal);
    
    cl::Event compArrMaxReduction(cl::Buffer& buf_in, int curr_size, bool absVal,
    vector<cl::Event> vecPrev);
    
    cl::Event compAvgInAndOut(cl::Image3D& imp_mask, cl::Image3D& img_in, vector<cl::Event> vecPrevEvents, bool useAllBands);
    
	// Here we are using width*height*depth*8
    cl::Image3DGL img_in_gl;//This is the object that holds the 3D texture of the nii image
    cl::Image2DGL img_phi_gl;//Output 
    cl::Image3D img_in;
    cl::Image3D img_phi;
    cl::Image3D img_mask;
    cl::Image3D img_curv_F;
    cl::Image3D img_dphidt;
    cl::Image3D img_newphi;
    
	std::vector<cl::Memory> cl_textures;
    
	// Here we are using width*height*depth*4
    cl::Buffer buf_avg_in_out;
    cl::Buffer buf_img_in;
    cl::Buffer buf_F;
    cl::Buffer buf_max_dphidt;
	cl::Buffer buf_sdf;
	cl::Buffer buf_mask;
    
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
    
    int max_warp_size;
    float alpha;
    
    cl_int err;
    cl_int res;
    
    CLManager clMan;
    //    CLProfiler prof;    
    
    cl::Context* context;
    cl::CommandQueue* queue;
    cl::Program* program;
    
    // These events will hold all the required events to compute the AVG
    vector<cl::Event> vecEvPrevAvgInOut;
    vector<cl::Event> vecEvAvg;
    vector<cl::Event> vecEvCurvF;
    vector<cl::Event> vecEvMaxF;
    vector<cl::Event> vecEvDphiDt;
    vector<cl::Event> vecEvMaxDphiDt;
    vector<cl::Event> vecEvNewPhi;
    vector<cl::Event> vecEvSmPhi;
    vector<cl::Event> vecEvSDF;
    
	cl::Event evAcOGL; //Event to aquire the input texture from OpenGL
    cl::Event evImgInWrt; //Event for writing the 'in' image
    cl::Event evImgSegWrt; //Wevent for writing the 'mask' image
    cl::Event evSDF;
    cl::Event evAvgInOut;
    cl::Event evCurvAndF;
    cl::Event evMaxF;
    cl::Event evDphiDt;
    cl::Event evMaxDphiDt;
    cl::Event evNewPhi;
    cl::Event evSmoothPhi;
    
    int currIter; //Current iteration of the algorithm
    int totalIterations;
    
    char* outputFile; //Final png file with the segmentation
};

#endif	/* ACTIVECONTOURS_H */

