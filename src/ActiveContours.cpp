/* 
 * File:   ActiveContours.cpp
 * Author: olmozavala
 * 
 * Created on October 10, 2011, 10:08 AM
 */

#define MAXF 1
#define MAXDPHIDT 2

// Writes is used to write results to disk
#define WRITE true
// PRINT_IMG_VAL is used to print images values (only for very small images)
#define PRINT_IMG_VAL false 

#define ITER 50 //Defines every how many iterations will write the outputs

#include "ActiveContours.h"
#include "SignedDistFunc.h"
#include "debug.h"

#include <sstream>
#include <iomanip>
#include <qt4/QtCore/qglobal.h>

// Inline function to cast char* to std:string
template <class T>
inline std::string to_string(char* prev, const T& t, char* post) {
    std::stringstream ss;
    ss << prev << t << post;
    return ss.str();
}

// Inline function to append to char* into one std:string
inline std::string appendStr(char* prev, char* post) {
    std::stringstream ss;
    ss << prev << post;
    return ss.str();
}

/**
 *  Constructor of the ActiveCountours class. It mainly
 * Initializes some variables. 
 */ 
ActiveContours::ActiveContours() {
    grp_size_x = 0;
    grp_size_y = 0;
	
    tot_grps_x = 0;
    tot_grps_y = 0;
}

ActiveContours::ActiveContours(const ActiveContours& orig) {
}

ActiveContours::~ActiveContours() {
}

/**
 * Initializes all the images objects. It receives the two textures pointers
 * from OpenGL.
 * @param {GLuint} tbo_in OpenGL texture pointer to the input image
 * @param {GLuint} tbo_out OpenGL texture pointer to the output maks or segmentation.
 */
void ActiveContours::initImagesArraysAndBuffers(GLuint& tbo_in, GLuint& tbo_out, 
		int locwidth, int locheight, int locdepth){
	
	//Sets the dimensions of the 3D texture
    width = locwidth;
    height = locheight;
    depth = locdepth;
	
    origin = CLManager::getSizeT(0, 0, 0);
    region = CLManager::getSizeT(width, height, 1);
	
    this->currIter = 0;
	
    try {
		
        clMan.getDeviceInfo(0); // Prints device info
		
		//Max number of work items per group
        dev_max_work_items = clMan.getMaxWorkGroupSize(0);
		// Max float elements of a local memory (per group)
		dev_max_local_mem = (int)clMan.getLocalMemSize(0)/(sizeof(float));

        CLManager::getGroupSize3D(dev_max_work_items, width, height, depth, grp_size_x, 
				grp_size_y, grp_size_z, tot_grps_x, tot_grps_y, tot_grps_z, true);
		
        cl::Context* context = clMan.getContext();
		
		//Maps the 3D texture of OpenGL to a cl::Image3DGL object 
		img_in_gl = cl::Image3DGL(*context, CL_MEM_READ_WRITE, GL_TEXTURE_3D, 0, tbo_in, &err);
		img_phi_gl = cl::Image3DGL(*context, CL_MEM_READ_WRITE, GL_TEXTURE_3D, 0, tbo_out, &err);
		
		//Add the images to the vector of cl::Memory that is used to acquire and release ogl objects
		cl_textures.push_back(img_in_gl);
		cl_textures.push_back(img_phi_gl);
		
		buf_avg_in_out = cl::Buffer(*context, CL_MEM_READ_WRITE,
                (size_t) 6* sizeof (float), NULL, &err);
		
		buf_size = width*height*depth;

        // Contains the values of F 
        buf_F = cl::Buffer(*context, CL_MEM_READ_WRITE, (size_t) buf_size * sizeof (float), NULL, &err);
		// Maximum value of F
        buf_max_F = cl::Buffer(*context, CL_MEM_READ_WRITE, (size_t) 1 * sizeof (float), NULL, &err);
		
        // Used to compute the maximum value of dphidt
        buf_max_dphidt = cl::Buffer(*context, CL_MEM_READ_WRITE, (size_t) 1 * sizeof (float), NULL, &err);
		
        buf_dphidt = cl::Buffer(*context, CL_MEM_READ_WRITE, (size_t) buf_size* sizeof (float), NULL, &err);

        buf_mask = cl::Buffer(*context, CL_MEM_READ_WRITE, buf_size * sizeof (unsigned int), NULL, &err);
		
        buf_phi = cl::Buffer(*context, CL_MEM_READ_WRITE,  buf_size * sizeof (float), NULL, &err);
		
        buf_smooth_phi = cl::Buffer(*context, CL_MEM_READ_WRITE,  buf_size * sizeof (float), NULL, &err);

        buf_img_in = cl::Buffer(*context, CL_MEM_READ_WRITE, buf_size * sizeof (float), NULL, &err);

        buf_curvature= cl::Buffer(*context, CL_MEM_READ_WRITE, buf_size * sizeof (float), NULL, &err);
		
        //--------------- Initialization of arrays ---------------------
        arr_img_out = new float[buf_size];
		
    } catch (cl::Error ex) {
        eout << "Error on initialization of images and buffers " << endl;
        clMan.printError(ex);
        return;
    }
}

void ActiveContours::loadProgram(int iter, float alpha, float dt) {
	
    this->totalIterations = iter;
    this->alpha = alpha;
    this->dt_smooth = dt;
    try {
        // Create the program from source
        clMan.initContext(true);
		clMan.addMultipleSources((char*) "src/resources/kernels/ActCount.cl", 
				(char*) "src/resources/kernels/SDFVoroBuf3D.cl");
        clMan.initQueue();
		
        context = clMan.getContext();
        queue = clMan.getQueue();
        program = clMan.getProgram();
		
    } catch (cl::Error ex) {
        clMan.printError(ex);
        return;
    }
}

/**
 * Runs the SDF from the initialized mask. 
 */
void ActiveContours::runSDF() {
	
	buf_size = width*height*depth;//Local variable of the buffer size

    try {
        queue->finish();//We need to finish everything that it was queued before
		
        //Writes the mask stored in 'arr_buf_mask' into the cl::Buffer 'buf_mask' 
        err = queue->enqueueWriteBuffer(buf_mask, CL_FALSE, 0, 
				sizeof (unsigned char)*buf_size, (void*) arr_buf_mask, 0, &evImgSegWrt);
		
        char* sdfPath = (char*) "images/SDF/"; //Path to save SDF images
		
//        if (WRITE) {//Writes the original mask
        if (false) {//Writes the original mask
			dout << "******** Writing original mask .... " << endl;
            string folder = appendStr(sdfPath, (char*) "OriginalMask/");
			ImageManager::write3DImageSDF( (char*) folder.c_str(), arr_buf_mask, width, height, depth);
		}
		
        SignedDistFunc sdfObj;
		
		dev_max_work_items = 512;
		dout << "Max warpsize for SDF: " << dev_max_work_items << endl;
        // It reads from buf_mask the mask, and writes into buf_phi the output
        evSDF_newPhi = sdfObj.run3DSDFBuf(&clMan, buf_mask, buf_phi, dev_max_work_items, width, 
				height, depth, evImgSegWrt, sdfPath);
		
//        if (WRITE) {// Saves the SDF result as an image
		if (WRITE) {
			cout << "--------------------Displaying some values of the SDF..." << endl;
			vecEvPrevPrinting.clear();
			vecEvPrevPrinting.push_back(evSDF_newPhi);
			//printBuffer(buf_phi, 10, vecEvPrevPrinting);
			printBuffer(buf_phi, 400, 0, width, height, vecEvPrevPrinting);
		}
        if (WRITE) {// Saves the SDF result as an image
            dout << "Saving SDF result..." << endl;
            string folder = appendStr(sdfPath, (char*) "SDFOutput/");
			
			vecEvPrevPrinting.push_back(evSDF_newPhi);
			//Reads from buf_phi (GPU) and writes to arr_img_out (Host)
            res = queue->enqueueReadBuffer(buf_phi, CL_TRUE, 0, sizeof (float) *buf_size, 
					(void*) arr_img_out, &vecEvPrevPrinting, 0);
			// Prints image into png file
			ImageManager::write3DImage((char*) folder.c_str(), arr_img_out, width, height, depth, 0);
        }
		
		vecEvPrevCurvature.push_back(evSDF_newPhi);

        // Copying result to GL 3d Texture
		err = queue->enqueueAcquireGLObjects(&cl_textures, NULL, &evAcOGL);
		queue->finish();

        dout << "Initializing origin and region with " << width << "," << height << "," << depth << endl;
        origin.push_back(0); origin.push_back(0); origin.push_back(0);
        region.push_back(width); region.push_back(height); region.push_back(depth);

        queue->enqueueCopyBufferToImage(buf_phi, img_phi_gl, (size_t)0, origin, 
                            region,  &vecEvPrevCurvature, &evAcOGL);

		err = queue->enqueueReleaseGLObjects(&cl_textures, NULL, 0);

        dout << "Out of ActiveContours initialization  SDF computed!......." << endl;
		
    } catch (cl::Error ex) {
        clMan.printError(ex);
        return;
    }
}

/**
 * Iterates the ACWE for several iterations using 1 or more bands
 * @param numIterations
 * @param useAllBands
 */
void ActiveContours::iterate(int numIterations, bool useAllBands) {

	//Default origin and region to copy the entire region of the 3D texture
	dout << "Initializing origin and region with " << width << "," << height << "," << depth << endl;
	origin.push_back(0); origin.push_back(0); origin.push_back(0);
	region.push_back(width); region.push_back(height); region.push_back(depth);
	
    cl::CommandQueue* queue = clMan.getQueue();
    try {
		err = queue->enqueueAcquireGLObjects(&cl_textures, NULL, &evAcOGL);
		queue->finish();
		
		if (currIter == 0) {
			//Copying img_in_gl to buf_img_in
			cl::Event evCopyInGlToIn;
			
			dout << "Copying input texture (img_in_gl) to cl_buffer buf_img_in" << endl;
			vecEvPrevTextToBuffer.push_back(evAcOGL);
			queue->enqueueCopyImageToBuffer(img_in_gl, buf_img_in, origin, 
								region, (size_t)0, &vecEvPrevTextToBuffer,&evCopyInGlToIn);

			if (WRITE) {//Writes the init image on the temporal folder
                // Sets the precision of cout to 2
                cout << std::setprecision(3) << endl;
				vecEvPrevPrinting.push_back(evCopyInGlToIn);
				res = queue->enqueueReadBuffer(buf_img_in, CL_TRUE, 0,
						sizeof (float) *width*height*depth, (void*) arr_img_out, &vecEvPrevPrinting, 0);
				
				bool normalized_values = 1;
				dout << "Done copying texture to buffer.... writing result to images/temp_results/InputImage/" << endl;
				ImageManager::write3DImage((char*) "images/temp_results/InputImage/",
						arr_img_out, width, height,depth, normalized_values);

				/* Just to test that the TEXTURE is being copied to img_in_gl correctly*/
				/*
				int rowSize = sizeof(float)*width;
				res = queue->enqueueReadImage(img_in_gl, CL_FALSE, origin, region, (size_t) rowSize ,
						(size_t)  (rowSize*height), (void*) arr_img_out, &vecEvPrevPrinting, 0);
				
				queue->finish(); //Finish everything before the iterations
				
				ImageManager::write3DImage((char*) "images/temp_results/3dTexture/",
						 arr_img_out, width, height,depth,  normalized_values);

				queue->finish(); //Finish everything before the iterations
				dout << "Writing done!!!!" << endl;
				*/
			}
			vecEvPrevAvgInOut.push_back(evCopyInGlToIn); //For the first iteration we need to wait to copy the texture
		}//If iter == 0

        //Compute the last iteration of this 'round'
        int lastIter = min(currIter + numIterations, totalIterations);
		
        // -------------------- MAIN Active Countours iteration
        for (; currIter < lastIter; currIter++) {
			
            if (currIter % ITER == 0) {
                cout << endl << endl << "******************** Iter " << currIter << " ******************** " << endl;
            }

            evAvgInOut_SmoothPhi = compAvgInAndOut(buf_phi, buf_img_in, vecEvPrevAvgInOut);

            if (WRITE) {// Prints the previous values of phi
                cout << endl << "----------- Previous Phi ------------" << endl;
				vecEvPrevPrinting.clear();
				vecEvPrevPrinting.push_back(evAvgInOut_SmoothPhi);
                printBuffer(buf_phi, 400, width*height*9, width, height, vecEvPrevPrinting);
                printBuffer(buf_phi, 400, width*height*10, width, height, vecEvPrevPrinting);
                printBuffer(buf_phi, 400, width*height*11, width, height, vecEvPrevPrinting);
			}		

            if (WRITE) {// Gets the final average values obtained
                cout << endl << "----------- Final Average  (avg out, avg in, count out, count in,  sum out, sum in)------------" << endl;
				vecEvPrevPrinting.clear();
				vecEvPrevPrinting.push_back(evAvgInOut_SmoothPhi);
				printBuffer(buf_avg_in_out, 6, vecEvPrevPrinting);
			}
			
            //It computes the curvatue and F values, the curvature is stored on the first layer
            //and the F values are stored on the second layer
            evCurvature_copySmoothToPhi = compCurvature(vecEvPrevCurvature);
			
            if (WRITE) {
                cout << "--------------------Displaying the value of curvature..." << endl;
				vecEvPrevPrinting.clear();
				vecEvPrevPrinting.push_back(evCurvature_copySmoothToPhi);
				//printBuffer(buf_curvature, 10, vecEvPrevPrinting);
                //printBuffer(buf_curvature, 400, 0, width, height, vecEvPrevPrinting);
                //printBuffer(buf_curvature, 400, width*height, width, height, vecEvPrevPrinting);
                //printBuffer(buf_curvature, 400, width*height*2, width, height, vecEvPrevPrinting);
                //printBuffer(buf_curvature, 400, width*height*3, width, height, vecEvPrevPrinting);
                //printBuffer(buf_curvature, 400, width*height*4, width, height, vecEvPrevPrinting);
                //printBuffer(buf_curvature, 400, width*height*5, width, height, vecEvPrevPrinting);
                //printBuffer(buf_curvature, 400, width*height*6, width, height, vecEvPrevPrinting);
                //printBuffer(buf_curvature, 400, width*height*7, width, height, vecEvPrevPrinting);
                //printBuffer(buf_curvature, 400, width*height*8, width, height, vecEvPrevPrinting);
                printBuffer(buf_curvature, 400, width*height*9, width, height, vecEvPrevPrinting);
                printBuffer(buf_curvature, 400, width*height*10, width, height, vecEvPrevPrinting);
                printBuffer(buf_curvature, 400, width*height*11, width, height, vecEvPrevPrinting);
                //printBuffer(buf_curvature, 400, width*height*12, width, height, vecEvPrevPrinting);
                //printBuffer(buf_curvature, 400, width*height*13, width, height, vecEvPrevPrinting);
                //printBuffer(buf_curvature, 400, width*height*14, width, height, vecEvPrevPrinting);
                //printBuffer(buf_curvature, 400, width*height*15, width, height, vecEvPrevPrinting);
                //printBuffer(buf_curvature, 400, width*height*16, width, height, vecEvPrevPrinting);
                //printBuffer(buf_curvature, 400, width*height*17, width, height, vecEvPrevPrinting);
                //printBuffer(buf_curvature, 400, width*height*18, width, height, vecEvPrevPrinting);
                //printBuffer(buf_curvature, 400, width*height*19, width, height, vecEvPrevPrinting);

            }
			
            // Computing the maximum F value (max value of:
            // pow( curr_img - avgIn, 2) - pow( curr_img - avgOut, 2))
            vecEvPrevF.push_back(evAvgInOut_SmoothPhi);//Wait to compute the average in and out
			evF = compF(vecEvPrevF);

            if (WRITE) {
                cout << "--------------------Displaying the value of F ..." << endl;
				vecEvPrevPrinting.clear();
				vecEvPrevPrinting.push_back(evF);
				//printBuffer(buf_F, 10, vecEvPrevPrinting);
                //printBuffer(buf_F, 400, 0, width, height, vecEvPrevPrinting);
                //printBuffer(buf_F, 400, width*height*7, width, height, vecEvPrevPrinting);
                //printBuffer(buf_F, 400, width*height*8, width, height, vecEvPrevPrinting);
                printBuffer(buf_F, 400, width*height*9, width, height, vecEvPrevPrinting);
                printBuffer(buf_F, 400, width*height*10, width, height, vecEvPrevPrinting);
                printBuffer(buf_F, 400, width*height*11, width, height, vecEvPrevPrinting);
            }

            //Computing maximum value of F
            vecEvPrevMaxF.push_back(evF);
            evMaxF = compReduce(buf_F, buf_max_F, true, vecEvPrevMaxF); // Use abs value 

            if (WRITE) {
                cout << "--------------------Displaying max value of F ..." << endl;
                vecEvPrevPrinting.clear();
                vecEvPrevPrinting.push_back(evF);
                printBuffer(buf_max_F, 1, vecEvPrevPrinting);
            }

            vecEvPrevDphiDt.push_back(evCurvature_copySmoothToPhi);// Wait for curvature
            vecEvPrevDphiDt.push_back(evMaxF);// Wait for max F -> and F
            evDphiDt_MaxDphiDt = compDphiDt(vecEvPrevDphiDt);

            if (WRITE) {
                cout << "--------------------Displaying values of Dphi/dt ..." << endl;
                vecEvPrevPrinting.clear();
                vecEvPrevPrinting.push_back(evDphiDt_MaxDphiDt);
                //printBuffer(buf_dphidt, 10, vecEvPrevPrinting);
                //printBuffer(buf_dphidt, 400, 0, width, height, vecEvPrevPrinting);
                //printBuffer(buf_dphidt, 400, width*height*1, width, height, vecEvPrevPrinting);
                //printBuffer(buf_dphidt, 400, width*height*2, width, height, vecEvPrevPrinting);
                //printBuffer(buf_dphidt, 400, width*height*3, width, height, vecEvPrevPrinting);
                //printBuffer(buf_dphidt, 400, width*height*4, width, height, vecEvPrevPrinting);
                //printBuffer(buf_dphidt, 400, width*height*5, width, height, vecEvPrevPrinting);
                //printBuffer(buf_dphidt, 400, width*height*6, width, height, vecEvPrevPrinting);
                //printBuffer(buf_dphidt, 400, width*height*7, width, height, vecEvPrevPrinting);
                //printBuffer(buf_dphidt, 400, width*height*8, width, height, vecEvPrevPrinting);
                printBuffer(buf_dphidt, 400, width*height*9, width, height, vecEvPrevPrinting);
                printBuffer(buf_dphidt, 400, width*height*10, width, height, vecEvPrevPrinting);
                printBuffer(buf_dphidt, 400, width*height*11, width, height, vecEvPrevPrinting);
                //printBuffer(buf_dphidt, 400, width*height*12, width, height, vecEvPrevPrinting);
                //printBuffer(buf_dphidt, 400, width*height*13, width, height, vecEvPrevPrinting);
                //printBuffer(buf_dphidt, 400, width*height*14, width, height, vecEvPrevPrinting);
                //printBuffer(buf_dphidt, 400, width*height*15, width, height, vecEvPrevPrinting);
                //printBuffer(buf_dphidt, 400, width*height*16, width, height, vecEvPrevPrinting);
                //printBuffer(buf_dphidt, 400, width*height*17, width, height, vecEvPrevPrinting);
                //printBuffer(buf_dphidt, 400, width*height*18, width, height, vecEvPrevPrinting);
                //printBuffer(buf_dphidt, 400, width*height*19, width, height, vecEvPrevPrinting);
            }

            vecEvPrevMaxDphiDt.push_back(evDphiDt_MaxDphiDt);
            evDphiDt_MaxDphiDt = compReduce(buf_dphidt, buf_max_dphidt, false, vecEvPrevMaxDphiDt ); 

            if (WRITE) {
                cout << "--------------------Displaying Max Dphi/dt ..." << endl;
                vecEvPrevPrinting.clear();
                vecEvPrevPrinting.push_back(evDphiDt_MaxDphiDt);
                printBuffer(buf_max_dphidt, 1, vecEvPrevPrinting);
            }

            vecEvPrevNewPhi.push_back(evDphiDt_MaxDphiDt);
            evSDF_newPhi = compNewPhi(vecEvPrevNewPhi); //This phi without smooth term

            if (WRITE) {
                cout << "--------------------Displaying values of new phi ..." << endl;
                vecEvPrevPrinting.clear();
                vecEvPrevPrinting.push_back(evSDF_newPhi);
                //printBuffer(buf_phi, 400, width*height*7, width, height, vecEvPrevPrinting);
//                printBuffer(buf_phi, 400, width*height*1, width, height, vecEvPrevPrinting);
//                printBuffer(buf_phi, 400, width*height*2, width, height, vecEvPrevPrinting);
//                printBuffer(buf_phi, 400, width*height*3, width, height, vecEvPrevPrinting);
//                printBuffer(buf_phi, 400, width*height*4, width, height, vecEvPrevPrinting);
//                printBuffer(buf_phi, 400, width*height*5, width, height, vecEvPrevPrinting);
//                printBuffer(buf_phi, 400, width*height*6, width, height, vecEvPrevPrinting);
                //printBuffer(buf_phi, 400, width*height*7, width, height, vecEvPrevPrinting);
                //printBuffer(buf_phi, 400, width*height*8, width, height, vecEvPrevPrinting);
                printBuffer(buf_phi, 400, width*height*9, width, height, vecEvPrevPrinting);
                printBuffer(buf_phi, 400, width*height*10, width, height, vecEvPrevPrinting);
                printBuffer(buf_phi, 400, width*height*11, width, height, vecEvPrevPrinting);
//                printBuffer(buf_phi, 400, width*height*12, width, height, vecEvPrevPrinting);
//                printBuffer(buf_phi, 400, width*height*13, width, height, vecEvPrevPrinting);
//                printBuffer(buf_phi, 400, width*height*14, width, height, vecEvPrevPrinting);
//                printBuffer(buf_phi, 400, width*height*15, width, height, vecEvPrevPrinting);
//                printBuffer(buf_phi, 400, width*height*16, width, height, vecEvPrevPrinting);
//                printBuffer(buf_phi, 400, width*height*17, width, height, vecEvPrevPrinting);
//                printBuffer(buf_phi, 400, width*height*18, width, height, vecEvPrevPrinting);
//                printBuffer(buf_phi, 400, width*height*19, width, height, vecEvPrevPrinting);
               
            }

            vecEvPrevSmPhi.push_back(evSDF_newPhi);
            evAvgInOut_SmoothPhi = smoothPhi(vecEvPrevSmPhi, dt_smooth); //This phi without smooth term

            if (WRITE) {
                cout << "--------------------Displaying values of smoothed phi ..." << endl;
                vecEvPrevPrinting.clear();
                vecEvPrevPrinting.push_back(evAvgInOut_SmoothPhi);
                //printBuffer(buf_smooth_phi, 400, width*height*7, width, height, vecEvPrevPrinting);
                //printBuffer(buf_smooth_phi, 400, width*height*1, width, height, vecEvPrevPrinting);
                //printBuffer(buf_smooth_phi, 400, width*height*2, width, height, vecEvPrevPrinting);
                //printBuffer(buf_smooth_phi, 400, width*height*3, width, height, vecEvPrevPrinting);
                //printBuffer(buf_smooth_phi, 400, width*height*4, width, height, vecEvPrevPrinting);
                //printBuffer(buf_smooth_phi, 400, width*height*5, width, height, vecEvPrevPrinting);
                //printBuffer(buf_smooth_phi, 400, width*height*6, width, height, vecEvPrevPrinting);
                //printBuffer(buf_smooth_phi, 400, width*height*7, width, height, vecEvPrevPrinting);
                //printBuffer(buf_smooth_phi, 400, width*height*8, width, height, vecEvPrevPrinting);
                printBuffer(buf_smooth_phi, 400, width*height*9, width, height, vecEvPrevPrinting);
                printBuffer(buf_smooth_phi, 400, width*height*10, width, height, vecEvPrevPrinting);
                printBuffer(buf_smooth_phi, 400, width*height*11, width, height, vecEvPrevPrinting);
                //printBuffer(buf_smooth_phi, 400, width*height*12, width, height, vecEvPrevPrinting);
                //printBuffer(buf_smooth_phi, 400, width*height*13, width, height, vecEvPrevPrinting);
                //printBuffer(buf_smooth_phi, 400, width*height*14, width, height, vecEvPrevPrinting);
                //printBuffer(buf_smooth_phi, 400, width*height*15, width, height, vecEvPrevPrinting);
                //printBuffer(buf_smooth_phi, 400, width*height*16, width, height, vecEvPrevPrinting);
                //printBuffer(buf_smooth_phi, 400, width*height*17, width, height, vecEvPrevPrinting);
                //printBuffer(buf_smooth_phi, 400, width*height*18, width, height, vecEvPrevPrinting);
                //printBuffer(buf_smooth_phi, 400, width*height*19, width, height, vecEvPrevPrinting);
            }

            vecEvPrevCopySmoothToPhi.push_back(evAvgInOut_SmoothPhi);
            res = queue->enqueueCopyBuffer(buf_smooth_phi, buf_phi,
                    (size_t)0, (size_t) 0,  (size_t) sizeof (float) *buf_size,
                    &vecEvPrevCopySmoothToPhi, &evCurvature_copySmoothToPhi);

            vecEvPrevAvgInOut.push_back(evCurvature_copySmoothToPhi);

            vecEvPrevAvgInOut.clear();
            vecEvPrevAvg.clear();
            vecEvPrevCurvature.clear();
            vecEvPrevF.clear();
            vecEvPrevMaxF.clear();
            vecEvPrevDphiDt.clear();
            vecEvPrevMaxDphiDt.clear();
            vecEvPrevNewPhi.clear();
            vecEvPrevSmPhi.clear();
            vecEvPrevSDF.clear();
            vecEvPrevPrinting.clear();
            vecEvPrevTextToBuffer.clear();
        }//Main loop

        queue->finish(); //Be sure we finish everything
        dout << "Done ..................." << endl;

        if (WRITE) {
            cout << "--------------------Writing new PHI as images in images/temp_results/newPhi/" << endl;

            vecEvPrevPrinting.push_back(evAvgInOut_SmoothPhi);
            //Reads from buf_phi (GPU) and writes to arr_img_out (Host)
            res = queue->enqueueReadBuffer(buf_smooth_phi, CL_TRUE, 0, sizeof (float) *buf_size, 
                    (void*) arr_img_out, &vecEvPrevPrinting, 0);
            // Prints image into png file
            ImageManager::write3DImage((char*) "images/temp_results/newPhi/", arr_img_out, width, height, depth, 0);
        }

        dout << " Copying back everything to OpenGL ... " << endl;
        vecEvPrevCopyPhiBackToGL.push_back(evAvgInOut_SmoothPhi);
        queue->enqueueCopyBufferToImage(buf_smooth_phi, img_phi_gl, (size_t)0, origin, 
                region, &vecEvPrevCopyPhiBackToGL, &evAcOGL);
        queue->finish(); //Be sure we finish everything
        err = queue->enqueueReleaseGLObjects(&cl_textures, NULL, 0);

    } catch (cl::Error ex) {
        cout << "EXCEPTION" << endl;
        clMan.printError(ex);
        return;
    }
}

cl::Event ActiveContours::copyBufToImg(cl::Buffer& buf, cl::Image3D& img, vector<cl::Event> vecEvPrev, bool bufHasAllBands) {
    if (WRITE) {
        cout << "----------- Copying buffer to image------------" << endl;
    }
    cl::Event evBufToText;
    try {
        cl::CommandQueue* queue = clMan.getQueue();
        cl::Program* program = clMan.getProgram();

        cl::Kernel kernelBufToText(*program, (char*) "bufToText");
        kernelBufToText.setArg(0, buf);
        kernelBufToText.setArg(1, img);
        kernelBufToText.setArg(2, width);
        kernelBufToText.setArg(3, height);
        //If false, then we assume the buffer only contains one band and we copy
        //it to the rest of the channels.
        kernelBufToText.setArg(4, (int)bufHasAllBands);

        // Do the work
        queue->enqueueNDRangeKernel(
                kernelBufToText,
                cl::NullRange,
                cl::NDRange((size_t) width, (size_t) height),
                cl::NDRange((size_t) grp_size_x, (size_t) grp_size_y),
                &vecEvPrev,
                &evBufToText);

    } catch (cl::Error ex) {
        clMan.printError(ex);
    }
    return evBufToText;
}

/**
 * This function simply prints the current NDRanges that are being used
 */
void ActiveContours::printNDRanges(int cols, int rows, int z, 
        int grp_cols, int grp_rows, int grp_z){
    dout << "NDRange = " << cols << " \t " << rows << " \t " << z << endl;
    dout << "NDGrps= " << grp_cols << " \t " << grp_rows << " \t " <<  grp_z << endl;
}
/**
 * Copies a 3D texture into a buffer
 */
cl::Event ActiveContours::copyTextToBuf(cl::Image3DGL& text, cl::Buffer& buf, vector<cl::Event> vecEvPrev) {

    cl::Event evBufToText;
    try {
        cl::CommandQueue* queue = clMan.getQueue();
        cl::Program* program = clMan.getProgram();

        cl::Kernel kernelBufToText(*program, (char*) "textToBuf");
        kernelBufToText.setArg(0, text);
        kernelBufToText.setArg(1, buf);

        dout << "Copying 3D Image to buffer (copyTextToBuf)" << endl;

        while(buf_size % dev_max_work_items != 0){
            dev_max_work_items--;
        }

        printNDRanges(buf_size, 0, 0, dev_max_work_items, 0, 0);

        queue->enqueueNDRangeKernel(
                kernelBufToText,
                cl::NullRange,
                cl::NDRange((size_t) buf_size),
                cl::NDRange((size_t) dev_max_work_items),
                &vecEvPrev,
                &evBufToText);

    } catch (cl::Error ex) {
        clMan.printError(ex);
    }
    return evBufToText;
}

/**
 * This method obtains the average pixel value of the input image for the pixels
 * 'outside' and 'inside' the mask.
 * @param buf_phi cl::Buffer& Reference to the SDF buffer
 * @param buf_img_in  cl::Buffer& Reference to the input image buffer
 * @param vecPrevEvents vector<cl::Event> List of events that need to be finished before start
 * @return cl::Event Last event required to compute the averages.
 */
cl::Event ActiveContours::compAvgInAndOut(cl::Buffer& buf_phi, cl::Buffer& buf_img_in,
        vector<cl::Event> vecPrevEvents) {

    dout << "Computing average pixel value for Phi > 0 and Phi < 0 ......" << endl;

    cl::Event evAvgInOut_SmoothPhi;
    try {
        // We are doing the average using just one Group. 
        // The threads on that group will change the memory they use to compute the avg

        //If we want to fill the local memory how many cells can a work item compute
        int cellsPerWorkItem = floor(dev_max_local_mem/dev_max_work_items);

        cl::CommandQueue* queue = clMan.getQueue();
        cl::Program* program = clMan.getProgram();

        cl::Kernel kernelAVGcolor(*program, (char*) "avgInOut");
        kernelAVGcolor.setArg(0, buf_phi);//SDF
        kernelAVGcolor.setArg(1, buf_img_in);//Texture data
        kernelAVGcolor.setArg(2, buf_avg_in_out);//Output variable
        kernelAVGcolor.setArg(3, buf_size);//Total number of values 
        kernelAVGcolor.setArg(4, cellsPerWorkItem);//Cells to compute in each iteration

        //We need to create an artificial increased size to get perfect fit
        int incSize = dev_max_work_items*(ceil(buf_size/dev_max_work_items));

        dout << "Max work items per group: " << dev_max_work_items << endl; 
        dout << "Size: " << buf_size << endl; 
        dout << "incSize: " << incSize << endl; 
        dout << "Cells to compute per work item:" << cellsPerWorkItem << endl;
        dout << "buf_avg_in_out size: " << 4 << endl;

        queue->enqueueNDRangeKernel(
                kernelAVGcolor,
                cl::NullRange,
                cl::NDRange((size_t) dev_max_work_items),
                cl::NDRange((size_t) dev_max_work_items),
                &vecPrevEvents,
                &evAvgInOut_SmoothPhi);

    } catch (cl::Error ex) {
        cout << "Error at compAvgInandOut" << endl;
        clMan.printError(ex);
    }
    return evAvgInOut_SmoothPhi;
}

cl::Event ActiveContours::smoothPhi(vector<cl::Event> vecEvPrev, float dt_smooth) {
    if (WRITE) {
        cout << "----------- Smoothing new phi ------------" << endl;
        cout << "----------- Using dt = " << dt_smooth << "------------" << endl;
    }
    cl::Event evSmooth;
    try {
        cl::CommandQueue* queue = clMan.getQueue();
        cl::Program* program = clMan.getProgram();

        cl::Kernel kernelSmoothPhi(*program, (char*) "smoothPhi");
        kernelSmoothPhi.setArg(0, buf_phi);
        kernelSmoothPhi.setArg(1, buf_smooth_phi);
        kernelSmoothPhi.setArg(2, dt_smooth);
        kernelSmoothPhi.setArg(3, width);
        kernelSmoothPhi.setArg(4, height);
        kernelSmoothPhi.setArg(5, depth);

        int threads = height*depth;
        int threadsPerGroup = dev_max_work_items;
        while( threads % threadsPerGroup != 0){
            threadsPerGroup --;
        }
        dout << "Threads: " << threads << " Threads per group: " << threadsPerGroup << endl;

        // Do the work
        queue->enqueueNDRangeKernel(
                kernelSmoothPhi,
                cl::NullRange,
                cl::NDRange((size_t) threads),
                cl::NDRange((size_t) threadsPerGroup),
                &vecEvPrev,
                &evSmooth);

    } catch (cl::Error ex) {
        clMan.printError(ex);
    }
    return evSmooth;
}

cl::Event ActiveContours::compNewPhi(vector<cl::Event> vecEvPrev) {
    if (WRITE) {
        cout << "----------- Computing new phi ------------" << endl;
    }
    cl::Event evSDF_newPhi;
    try {
        cl::CommandQueue* queue = clMan.getQueue();
        cl::Program* program = clMan.getProgram();

        cl::Kernel kernelDphiDt(*program, (char*) "newphi");
        kernelDphiDt.setArg(0, buf_phi);
        kernelDphiDt.setArg(1, buf_dphidt);
        kernelDphiDt.setArg(2, buf_max_dphidt);
        kernelDphiDt.setArg(3, width);
        kernelDphiDt.setArg(4, height);

        int threads = height*depth;
        int threadsPerGroup = dev_max_work_items;
        while( threads % threadsPerGroup != 0){
            threadsPerGroup --;
        }
        dout << "Threads: " << threads << " Threads per group: " << threadsPerGroup << endl;

        // Do the work
        queue->enqueueNDRangeKernel(
                kernelDphiDt,
                cl::NullRange,
                cl::NDRange((size_t) threads),
                cl::NDRange((size_t) threadsPerGroup),
                &vecEvPrev,
                &evSDF_newPhi);

    } catch (cl::Error ex) {
        clMan.printError(ex);
    }
    return evSDF_newPhi;
}

cl::Event ActiveContours::compDphiDt(vector<cl::Event> vecEvPrev) {
    cl::Event evCompDphiDt;
    try {
        cl::CommandQueue* queue = clMan.getQueue();
        cl::Program* program = clMan.getProgram();

        cl::Kernel kernelDphiDt(*program, (char*) "dphidt");
        kernelDphiDt.setArg(0, buf_curvature);
        kernelDphiDt.setArg(1, buf_F);
        kernelDphiDt.setArg(2, buf_max_F);
        kernelDphiDt.setArg(3, buf_dphidt);
        kernelDphiDt.setArg(4, alpha);
        kernelDphiDt.setArg(5, width);
        kernelDphiDt.setArg(6, height);

        int threads = height*depth;
        int threadsPerGroup = dev_max_work_items;
        while( threads % threadsPerGroup != 0){
            threadsPerGroup --;
        }
        dout << "Threads: " << threads << " Threads per group: " << threadsPerGroup << endl;

        queue->enqueueNDRangeKernel(
                kernelDphiDt,
                cl::NullRange,
                cl::NDRange((size_t) threads),
                cl::NDRange((size_t) threadsPerGroup),
                &vecEvPrev,
                &evCompDphiDt);

    } catch (cl::Error ex) {
        clMan.printError(ex);
    }
    return evCompDphiDt;
}

/**
 * Computes the curvature of phi (buf_phi). Each thread computes a row of the image
 * Input buf_phi
 * Output buf_curvature
 * @param vecEvPrev Events that are required to finish befor computing the curvature 
 * 					It depends on the previous computation of buf_phi
 * @return 
 */
cl::Event ActiveContours::compCurvature(vector<cl::Event> vecEvPrev) {

    cl::Event evCurvF;
    dout << endl << " ----------------- Computing Curvature ---------" << endl;

    try {
        cl::CommandQueue* queue = clMan.getQueue();
        cl::Program* program = clMan.getProgram();

        cl::Kernel kernelCurvature(*program, (char*) "curvature");
        kernelCurvature.setArg(0, buf_phi);
        kernelCurvature.setArg(1, buf_curvature);
        kernelCurvature.setArg(2, width);
        kernelCurvature.setArg(3, height);
        kernelCurvature.setArg(4, depth);

        int threads = height*depth;
        int threadsPerGroup = dev_max_work_items;
        while( threads % threadsPerGroup != 0){
            threadsPerGroup --;
        }
        dout << "Threads: " << threads << " Threads per group: " << threadsPerGroup << endl;

        queue->enqueueNDRangeKernel(
                kernelCurvature,
                cl::NullRange,
                cl::NDRange((size_t) threads),
                cl::NDRange((size_t) threadsPerGroup),
                &vecEvPrev,
                &evCurvF);

    } catch (cl::Error ex) {
        clMan.printError(ex);
    }
    return evCurvF;
}

/**
 * Computes the force of image formation (I-u)^2 - (I-v)^2
 * @param vecEvPrev Depends on the computation of the average in and out
 * @return 
 */
cl::Event ActiveContours::compF(vector<cl::Event> vecEvPrev) {

    cl::Event evCompF;
    dout << endl << " ----------------- Computing F ---------" << endl;

    try {
        cl::CommandQueue* queue = clMan.getQueue();
        cl::Program* program = clMan.getProgram();

        cl::Kernel kernelCurvature(*program, (char*) "compF");
        kernelCurvature.setArg(0, buf_avg_in_out);
        kernelCurvature.setArg(1, buf_img_in);
        kernelCurvature.setArg(2, buf_F);
        kernelCurvature.setArg(3, width);
        kernelCurvature.setArg(4, height);
        kernelCurvature.setArg(5, depth);

        int threads = height*depth;
        int threadsPerGroup = dev_max_work_items;
        while( threads % threadsPerGroup != 0){
            threadsPerGroup --;
        }
        dout << "Threads: " << threads << " Threads per group: " << threadsPerGroup << endl;

        queue->enqueueNDRangeKernel(
                kernelCurvature,
                cl::NullRange,
                cl::NDRange((size_t) threads),
                cl::NDRange((size_t) threadsPerGroup),
                &vecEvPrev,
                &evCompF);

    } catch (cl::Error ex) {
        clMan.printError(ex);
    }
    return evCompF;
}

/**
 * Compute the maximum value of a 2D image. (The buffer should already be initialized)
 * It finds the maximum for every column, and then the maximum in general
 * @param img
 * @param vecEvPrev
 * @param buf_out
 * @param layer
 * @param absVal
 * @return 
 */
cl::Event ActiveContours::compReduce(cl::Buffer& buf, cl::Buffer& buf_out, 
        bool absVal, vector<cl::Event> vecEvPrev) {

    if (WRITE) {
        cout << endl << "--------- Computing Reduce -------------" << endl;
    }

    cl::Event evReduce;
    try {
        cl::CommandQueue* queue = clMan.getQueue();
        cl::Program* program = clMan.getProgram();

        int length = width*height*depth;
        int threadsPerGroup = min(dev_max_work_items,length);
        int threads = threadsPerGroup;//We only want one group

        dout << "Threads: " << threads << 
            " Threads per group: " << threadsPerGroup << 
            " Length: " << length << endl;

        cl::Kernel kernelMaxValue(*program, (char*) "reduce");

        kernelMaxValue.setArg(0, buf);
        kernelMaxValue.setArg(1, sizeof (float) * threadsPerGroup, NULL);
        kernelMaxValue.setArg(2, buf_out);
        kernelMaxValue.setArg(3, length);
        kernelMaxValue.setArg(4, absVal ? 1 : 0); //Indicates if we take the absolute vale or not

        // Do the work
        queue->enqueueNDRangeKernel(
                kernelMaxValue,
                cl::NullRange,
                cl::NDRange((size_t) threads),
                cl::NDRange((size_t) threadsPerGroup),
                &vecEvPrev,
                &evReduce);

    } catch (cl::Error ex) {
        clMan.printError(ex);
    }

    return evReduce;
}

/**
 * Creates a 'cube' mask from a 3D space. The cube is filled with
 * ones and the rest is filled with 0's
 * @param width
 * @param height
 * @param depth
 * @param rowStart X start position of the cube 
 * @param rowEnd Y start position of the cube 
 * @param colStart Y start position of the cube 
 * @param colEnd Y end position of the cube 
 * @param depthStart Z start position of the cube 
 * @param depthEnd Z end position of the cube 
 */
void ActiveContours::create3DMask(int width, int height, int depth,
        int colStart, int colEnd, int rowStart, int rowEnd, int depthStart, int depthEnd) {

    arr_buf_mask = new unsigned char[width * height * depth];

    //Update local width and height of the image
    width = width;
    height = height;
    depth  = depth;

    int size = width * height * depth;
    int indx = 0;

    cout << "--------------------- Creating mask (" << width << "," 
        << height << "," << depth << ") ----------" << endl;

    dout << "col min: " << colStart << " col max: " << colEnd << endl;
    dout << "row min: " << rowStart << " row max: " << rowEnd << endl;
    dout << "depth min: " << depthStart << " depth max: " << depthEnd << endl;

    //Initialize to 0
    for (int i = 0; i < size; i++) {
        arr_buf_mask[i] = 0; // Red value
    }

    int count = 0;
    //Set the internal mask to 1
    for (int z = depthStart; z < depthEnd; z++) {
        for (int row = rowStart; row < rowEnd; row++) {
            //indx = ImageManager::indxFromCoord3D(width, height, row, rowStart, z);
            indx = width * height * z + width * row + colStart;
            for (int col = colStart; col < colEnd; col++) {
                arr_buf_mask[indx] = 1; //R
                indx = indx + 1;
                count++;
            }
        }
    }

    dout << "Total one's on mask: " << count << endl;
}


void ActiveContours::printBuffer(cl::Buffer& buf, int size, int offset, int width, int height, vector<cl::Event> vecPrev){

    cl::CommandQueue* queue = clMan.getQueue();

    float* result = new float[size];
    dout << "Reading: " << size<< " elements" << endl;
    // buffer, block, offset, size, ptr, ev_wait, new_ev
    res = queue->enqueueReadBuffer
        (buf, CL_TRUE, (size_t) (sizeof(float)*offset), (size_t) (sizeof(float)*size), 
         (void*) result, &vecPrev, 0);

    queue->finish();

    int count = 0;
    while(count < size){
        cout << "------ Slice ------" << (offset/(width*height) + 1) << endl << endl;
        for(int row= 0; row<height; row++){
            for(int col= 0; col<width; col++){
                if( count < size){
                    cout << result[count] << "\t";
                    count++;
                }else{
                    break;
                }
            }//col
            cout << endl;
            if( count >= size){
                break;
            }
        }//row
    }

} 


void ActiveContours::printBuffer(cl::Buffer& buf, int size, vector<cl::Event> vecPrev){

    cl::CommandQueue* queue = clMan.getQueue();

    float* result = new float[size];
    dout << "Reading: " << size<< " elements" << endl;
    res = queue->enqueueReadBuffer(buf,
            CL_TRUE, (size_t) 0, (size_t) (sizeof(float)*size),
            (void*) result, &vecPrev, 0);

    queue->finish();

    for(int i = 0; i< size; i++){
        dout << "Value at i: " << i << " = " << result[i] << endl;
    }

}
