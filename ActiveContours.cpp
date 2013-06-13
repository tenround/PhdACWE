/* 
 * File:   ActiveContours.cpp
 * Author: olmozavala
 * 
 * Created on October 10, 2011, 10:08 AM
 */

#define MAXF 1
#define MAXDPHIDT 2

// Writes is used to write results to disk
#define WRITE false 
//#define WRITE true 
// PRINT_IMG_VAL is used to print images values (only for very small images)
#define PRINT_IMG_VAL false 

#define ITER 50 //Defines every how many iterations will write the outputs

#include "ActiveContours.h"
#include "SignedDistFunc.h"
#include "debug.h"

#include <sstream>

template <class T>
inline std::string to_string(char* prev, const T& t, char* post) {
    std::stringstream ss;
    ss << prev << t << post;
    return ss.str();
}

inline std::string appendStr(char* prev, char* post) {
    std::stringstream ss;
    ss << prev << post;
    return ss.str();
}

ActiveContours::ActiveContours() {
    grp_size_x = 0;
    grp_size_y = 0;

    tot_grps_x = 0;
    tot_grps_y = 0;

    usingOGL = true; //By default we set true using openGL (not good)
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
void ActiveContours::initImagesArraysAndBuffers(GLuint& tbo_in, GLuint& tbo_out) {
    origin = CLManager::getSizeT(0, 0, 0);
    region = CLManager::getSizeT(width, height, 1);

    try {
        cl::Context* context = clMan.getContext();

        if (usingOGL) {//Assume we use OpenGL

            img_in_gl = cl::Image2DGL(*context, CL_MEM_READ_WRITE, GL_TEXTURE_2D, 0, tbo_in, &err);
            img_phi_gl = cl::Image2DGL(*context, CL_MEM_READ_WRITE, GL_TEXTURE_2D, 0, tbo_out, &err);

            //Add the images to the vector of cl::Memory that is used to acquire and release ogl objects
            cl_textures.push_back(img_in_gl);
            cl_textures.push_back(img_phi_gl);
        }

        img_phi = cl::Image2D(*context, CL_MEM_READ_WRITE,
                cl::ImageFormat(CL_RGBA, CL_FLOAT), (size_t) width, (size_t) height, 0, 0, &err);

        img_in = cl::Image2D(*context, CL_MEM_READ_WRITE,
                cl::ImageFormat(CL_RGBA, CL_FLOAT), (size_t) width, (size_t) height, 0, 0, &err);

        //Initial segmentation of the object, normally a square
        img_mask = cl::Image2D(*context, CL_MEM_READ_WRITE,
                cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8), (size_t) width, (size_t) height, 0, 0, &err);

        // Temporal image that holds the new level set segmentation (phi)
        img_newphi = cl::Image2D(*context, CL_MEM_READ_WRITE,
                cl::ImageFormat(CL_RGBA, CL_FLOAT), (size_t) width, (size_t) height, 0, 0, &err);

        img_curv_F = cl::Image2D(*context, CL_MEM_READ_WRITE,
                cl::ImageFormat(CL_RGBA, CL_FLOAT), (size_t) width, (size_t) height, 0, 0, &err);

        img_dphidt = cl::Image2D(*context, CL_MEM_READ_WRITE,
                cl::ImageFormat(CL_RGBA, CL_FLOAT), (size_t) width, (size_t) height, 0, 0, &err);


        dout << "------------- GROUPS -----" << endl;
        dout << "X = " << tot_grps_x << " Y " << tot_grps_y << endl;
        int defSize = tot_grps_x * tot_grps_y;

        //-------------------- Initialization of buffers --------------

        // This array will contain the average gray value inside and outside the object (at the end
        // this two values should be on the first two positions)
        buf_avg_in_out = cl::Buffer(*context, CL_MEM_READ_WRITE,
                (size_t) defSize * 2 * sizeof (float), NULL, &err);

        // Contains the values of F 
        buf_F = cl::Buffer(*context, CL_MEM_READ_WRITE, (size_t) defSize * sizeof (float), NULL, &err);

        // Used to compute the maximum value of dphidt
        buf_max_dphidt = cl::Buffer(*context, CL_MEM_READ_WRITE, (size_t) defSize * sizeof (float), NULL, &err);

        buf_mask = cl::Buffer(*context, CL_MEM_READ_WRITE, width * height * sizeof (unsigned int), NULL, &err);

        buf_sdf = cl::Buffer(*context, CL_MEM_READ_WRITE, 4 * width * height * sizeof (float), NULL, &err);

        buf_img_in = cl::Buffer(*context, CL_MEM_READ_WRITE, 4 * width * height * sizeof (float), NULL, &err);

        //--------------- Initialization of arrays ---------------------

        arr_img_out = new float[width * height * 4];
        arr_max_F = new float[defSize];
        arr_max_dphidt = new float[defSize];
        arr_avgInOutByGrp = new float[defSize * 2]; //Is double the normal size because we have in and out

    } catch (cl::Error ex) {
        eout << "Error on initialization of images and buffers " << endl;
        clMan.printError(ex);
        return;
    }
}

void ActiveContours::init(int SDFmethod, char* inputFile, char* outputFile,
        int iter, float alpha, float def_dt, int* maskPos) {
    usingOGL = false;
    this->loadProgram(SDFmethod, inputFile, outputFile, iter, alpha, def_dt,  -1, -1);
    createRGBAMask(width, height, maskPos[0], maskPos[1], maskPos[2], maskPos[3]);
}

void ActiveContours::loadProgram(int SDFmethod, char* inputFile, char* outputFile, int iter,
        float alpha, float dt, int locwidth, int locheight) {

    //Important! here is where we set the width and height
    width = locwidth;
    height = locheight;

    cout << "Delete. At loadProgram size is: (" << locwidth << "," << locheight << ")" << endl;
    cout << "Delete. At loadProgram size is: (" << width << "," << height << ")" << endl;

    this->outputFile = outputFile;
    this->totalIterations = iter;
    this->currIter = 0;
    this->alpha = alpha;
    this->SDFmethod = SDFmethod;

    try {
        // Create the program from source
        clMan.initContext(usingOGL);
        switch (SDFmethod) {
            case SDFOZ:
                clMan.addMultipleSources((char*) "ActCount.cl", (char*) "SDF.cl");
                break;
            case SDFVORO:
                clMan.addMultipleSources((char*) "ActCount.cl", (char*) "SDFVoroBuf.cl");
                break;
        }
        clMan.initQueue();

        context = clMan.getContext();
        queue = clMan.getQueue();
        program = clMan.getProgram();

        //If we are not receiving the width and height then we need to 
        //read the image size again
        if( (width == -1) || (height == -1)){
            BYTE* imgByte = ImageManager::loadImageByte(inputFile, width, height);
            arr_img_in = ImageManager::byteToFloat(imgByte, width * height * 4);
        }

        if (WRITE) {//Writes the init image on the temporal folder
            dout << "Saving Original image " << endl;
            ImageManager::writeImage((char*) "images/SDF/CurrentRun/Original.png", arr_img_in, FIF_PNG, width, height);
        }

        //createRGBAMask(width, height, maskPos[0], maskPos[1], maskPos[2], maskPos[3], arr_buf_mask);

        clMan.getDeviceInfo(0); // Prints device info

        max_warp_size = clMan.getMaxWorkGroupSize(0);
        //        max_warp_size = 64; //TODO not always work with the obtained max_warp_size (But I am not sure why)

        // Gets the group size from the with and height of the image and the mas number of threads per group
        CLManager::getGroupSize(max_warp_size, width, height, grp_size_x, grp_size_y, tot_grps_x, tot_grps_y, WRITE);
        cout << "Delete. At END of loadProgram size is: (" << width << "," << height << ")" << endl;
    } catch (cl::Error ex) {
        clMan.printError(ex);
        return;
    }
}

void ActiveContours::runSDF() {

    dout << endl << "--------Running SDF ("<< width << "," << height << ")" << endl;
    try {
        // Create the program from source
        if (!usingOGL) {
            //If we are not using the OpenGL texture, then we write it from the array.
            err = queue->enqueueWriteImage(img_in, CL_FALSE, origin, region, 0, 0, (void*) arr_img_in, 0, &evImgInWrt);
        }

        queue->finish();

        //Writes the mask as the original value for 
        err = queue->enqueueWriteBuffer(buf_mask, 
                CL_FALSE, 0, sizeof (float) *width*height, (void*) arr_buf_mask, 0, &evImgSegWrt);
        //        err = queue->enqueueWriteImage(img_mask, CL_FALSE, origin, region, 0, 0, (void*) arr_img_phi, 0, &evImgSegWrt);

        char* sdfPath = (char*) "images/SDF/CurrentRun/"; //Path to save SDF images

        if (WRITE) {//Writes the original mask
            string fileName = appendStr(sdfPath, (char*) "MASK.png");
            ImageManager::writeGrayScaleImage((char*) fileName.c_str(), arr_buf_mask, FIF_PNG, width, height);
            if (PRINT_IMG_VAL) {
                cout << "Mask: " << endl;
                ImageManager::printImage(width, height, arr_buf_mask, 1);
            }
            cout << "--------------------Computing the SDF..." << endl;
        }

        vecWriteImage.push_back(evImgSegWrt);

        SignedDistFunc sdfObj;

        //        It reads from img_mask the mask, and writes into img_phi the output
        evSDF = sdfObj.runSDFBuf(&clMan, SDFmethod, buf_mask, buf_sdf, max_warp_size, width, height, evImgSegWrt, sdfPath);
        vecEvSDF.push_back(evSDF);

        if (true) {// Saves the SDF result as an image
            cout << "Saving SDF result..." << endl;
            string fileName = appendStr(sdfPath, (char*) "SDFResult.png");

			//Reads from buf_sdf (GPU) and writes to arr_img_out (Host)
            res = queue->enqueueReadBuffer(buf_sdf, CL_TRUE, 0, sizeof (float) *width*height, (void*) arr_img_out, &vecEvSDF, 0);
			// Prints image into png file
            ImageManager::writeGrayScaleImage((char*) fileName.c_str(), arr_img_out, FIF_PNG, width, height);
            if (PRINT_IMG_VAL) {
                ImageManager::printImage(width, height, arr_img_out, 1);
            }
        }

        queue->finish(); //Be sure we finish
        dout << "Out of ActiveContours initialization ......." << endl;

    } catch (cl::Error ex) {
        clMan.printError(ex);
        return;
    }
    //delete[] arr_img_in;
    //delete[] arr_buf_mask;
}

void ActiveContours::iterate(int numIterations, bool useAllBands) {

    cl::CommandQueue* queue = clMan.getQueue();
    try {
        vector<cl::Event> vecCopyTextBuf;
        cl::Event evCopyBufToPhi;

        if (usingOGL) {
            //err = queue->enqueueAcquireGLObjects(&cl_textures, NULL, &evAcOGL);
            err = queue->enqueueAcquireGLObjects(&cl_textures, NULL, 0);
            vecEvPrevAvgInOut.push_back(evAcOGL);
            queue->finish();

            if (currIter == 0) {
                //Copying img_in_gl to img_in
                cl::Event evCopyInGlToIn;
                cl::Event evCopyBufToImg;

                evCopyInGlToIn = this->copyTextToBuf(img_in_gl, buf_img_in, vecCopyTextBuf);
                vecCopyTextBuf.push_back(evCopyInGlToIn);

                if (WRITE) {//Writes the init image on the temporal folder
                //if (true) {//Writes the init image on the temporal folder
                    res = queue->enqueueReadBuffer(buf_img_in, CL_TRUE, 0,
                            sizeof (float) *width*height*4, (void*) arr_img_out, &vecCopyTextBuf, 0);

                    cout << "Saving buffer image in" << endl;
                    ImageManager::writeImage((char*) "images/temp_results/InputImageAsBuffer.png",
																	arr_img_out, FIF_PNG, width, height);
                    if (PRINT_IMG_VAL) {
                        cout << "Printing Saving buffer image in" << endl;
                        ImageManager::printImage(width, height, arr_img_out, 1);
                    }
                }

                evCopyBufToImg = this->copyBufToImg(buf_img_in, img_in, vecCopyTextBuf, true);
                vecCopyTextBuf.push_back(evCopyBufToImg);
                cout << "After buf_img_in -> img_in" << endl;

                //Copying the result from the SDF to the initial phi
                evCopyBufToPhi = this->copyBufToImg(buf_sdf, img_phi, vecCopyTextBuf, false);
                vecCopyTextBuf.push_back(evCopyBufToPhi);
                cout << "After buf_sdf-> img_phi" << endl;

                if (WRITE) {
                    res = queue->enqueueReadImage(img_in,
                            CL_TRUE, origin, region, 0, 0, (void*) arr_img_out, &vecCopyTextBuf, 0);

                    if (PRINT_IMG_VAL) {
                        cout << "&&&&&     Printing buffer image in" << endl;
                        ImageManager::printImage(width, height, arr_img_out, 4);
                    }

                    res = queue->enqueueReadImage(img_phi,
                            CL_TRUE, origin, region, 0, 0, (void*) arr_img_out, &vecCopyTextBuf, 0);

                    if (PRINT_IMG_VAL) {
                        cout << "&&&&&     Printing SDF (img_phi)" << endl;
                        ImageManager::printImage(width, height, arr_img_out, 4);
                    }
                }
            }//If iter == 0

        }
        //Compute the last iteration of this 'round'
        int lastIter = min(currIter + numIterations, totalIterations);

        // -------------------- MAIN Active Countours iteration
        queue->finish(); //Finish everything before the iterations
        for (currIter; currIter < lastIter; currIter++) {

            if (currIter % ITER == 0) {
                cout << endl << endl << "******************** Iter " << currIter << " ******************** " << endl;
            }

            vector<cl::Event> prevEv;
            evAvgInOut = compAvgInAndOut(img_phi, img_in, prevEv, useAllBands);

            vecEvAvg.push_back(evAvgInOut);

            if (WRITE) {// Gets the final average values obtained
                cout << endl << "----------- Final Average  ------------" << endl;

                res = queue->enqueueReadBuffer(buf_avg_in_out,
                        CL_TRUE, (size_t) 0, (size_t) (sizeof (float) *2),
                        (void*) arr_avgInOutByGrp, &vecEvAvg, 0);

                cout << "Avg in =" << arr_avgInOutByGrp[0] << " out=" << arr_avgInOutByGrp[1] << endl;
            }

            //            It computes the curvatue and F values, the curvature is stored on the first layer
            //and the F values are stored on the second layer
            evCurvAndF = compCurvAndF(vecEvAvg, useAllBands);
            vecEvCurvF.push_back(evCurvAndF);

            if (WRITE) {
                cout << "--------------------Saving the value of F and curvature..." << endl;

                res = queue->enqueueReadImage(img_curv_F,
                        CL_TRUE, origin, region, 0, 0, (void*) arr_img_out, &vecEvCurvF, 0);

                string name = to_string<int>((char*) "images/temp_results/CurvAndF", currIter, (char*) ".png");
                ImageManager::writeImage((char*) name.c_str(), arr_img_out, FIF_PNG, width, height);
            }

            // Computing the maximum F value (max value of:
            // pow( curr_img - avgIn, 2) - pow( curr_img - avgOut, 2))
            // this is stored on layer number 2

            evMaxF = compImgMaxReduction(img_curv_F, vecEvCurvF,
                    buf_F, 2, true); // Use abs value 

            vecEvMaxF.push_back(evMaxF);

            if (WRITE) {
                float* result = new float[1];

                res = queue->enqueueReadBuffer(buf_F,
                        CL_TRUE, (size_t) 0, (size_t) (sizeof (float)),
                        (void*) result, &vecEvMaxF, 0);

                cout << " -- Final Max F value: " << (float) result[0] << endl;
                delete[] result;
            }

            evDphiDt = compDphiDt(vecEvMaxF);
            vecEvDphiDt.push_back(evDphiDt);

            if (WRITE) {
                res = queue->enqueueReadImage(img_dphidt,
                        CL_TRUE, origin, region, 0, 0, (void*) arr_img_out, &vecEvDphiDt);

                string name = to_string<int>((char*) "images/temp_results/dphidt", currIter, (char*) ".png");
                ImageManager::writeImage((char*) name.c_str(), arr_img_out, FIF_PNG, width, height);
            }

            //Computes maximum dphi/dt value
            evMaxDphiDt = compImgMaxReduction(img_dphidt, vecEvDphiDt,
                    buf_max_dphidt, 1, false); // Use abs value 

            vecEvMaxDphiDt.push_back(evMaxDphiDt);

            if (WRITE) {
                float* result = new float[1];

                res = queue->enqueueReadBuffer(buf_max_dphidt,
                        CL_TRUE, (size_t) 0, (size_t) (sizeof (float)),
                        (void*) result, &vecEvMaxDphiDt, 0);

                cout << " -- Final Max DphiDt Value: " << (float) result[0] << endl;
                delete[] result;
            }

            evNewPhi = compNewPhi(vecEvMaxDphiDt); //This phi without smooth term
            vecEvNewPhi.push_back(evNewPhi);

            if (WRITE) {
                res = queue->enqueueReadImage(img_newphi,
                        CL_TRUE, origin, region, 0, 0, (void*) arr_img_out,
                        &vecEvNewPhi, 0);

                string name = to_string<int>((char*) "images/temp_results/newphiwosmoothing", currIter, (char*) ".png");
                ImageManager::writeImage((char*) name.c_str(), arr_img_out, FIF_PNG, width, height);
            }


            evSmoothPhi = smoothPhi(vecEvNewPhi, alpha); //This phi without smooth term

            vecEvSmPhi.push_back(evSmoothPhi);

            //Print image every time it enters (remove at the end)
            if (WRITE) {
                if (currIter % ITER == 1) {
                    res = queue->enqueueReadImage(img_phi,
                            CL_TRUE, origin, region, 0, 0, (void*) arr_img_out,
                            &vecEvSmPhi, 0);

                    string name = to_string<int>((char*) "images/temp_results/newphi", currIter, (char*) ".png");
                    ImageManager::writeImage((char*) name.c_str(), arr_img_out, FIF_PNG, width, height);
                }
            }

            vecEvPrevAvgInOut.clear();
            vecEvAvg.clear();
            vecEvCurvF.clear();
            vecEvMaxF.clear();
            vecEvDphiDt.clear();
            vecEvMaxDphiDt.clear();
            vecEvNewPhi.clear();
            vecEvSmPhi.clear();

            vecEvPrevAvgInOut.push_back(evSmoothPhi);

        }//Main loop

        //Copying result to final texture
        queue->finish();
        copySegToText();

        //Only in the last iteration we save the final output and delete the arrays
//        if (currIter == totalIterations) {
        if (false){

            cout << "Last iteration, cleaning objects....... " << endl;
            res = queue->enqueueReadImage(img_phi,
                    CL_TRUE, origin, region, 0, 0, (void*) arr_img_out, &vecEvPrevAvgInOut, 0);

            ImageManager::writeImage((char*) outputFile, arr_img_out,
                    FIF_PNG, width, height);

            delete[] arr_img_phi;
            delete[] arr_buf_mask;
            delete[] arr_max_F;
            delete[] arr_avgInOutByGrp;
        }

        if (usingOGL) {
            queue->finish(); //Be sure we finish everything
            err = queue->enqueueReleaseGLObjects(&cl_textures, NULL, 0);
        }

    } catch (cl::Error ex) {
        cout << "EXCEPTION" << endl;
        clMan.printError(ex);
        return;
    }


}

/**
 * This function reads data from the current segmentation (img_phi)
 * and the input image (img_in_gl) and merge the values
 * to draw the contour of the segmentation into img_phi_gl
 * Depending on the 'threshold' is what we take as contour
 */
void ActiveContours::copySegToText() {
    if (WRITE) {
        cout << "----------- Copying current segmentation------------" << endl;
    }
    try {
        cl::CommandQueue* queue = clMan.getQueue();
        cl::Program* program = clMan.getProgram();

        cl::Kernel kernelBufToText(*program, (char*) "segmToTexture");
        kernelBufToText.setArg(0, img_phi);
        kernelBufToText.setArg(1, img_in_gl);
        kernelBufToText.setArg(2, img_phi_gl);

        // Do the work
        queue->enqueueNDRangeKernel(
                kernelBufToText,
                cl::NullRange,
                cl::NDRange((size_t) width, (size_t) height),
                cl::NDRange((size_t) grp_size_x, (size_t) grp_size_y),
                NULL,
                NULL);

    } catch (cl::Error ex) {
        clMan.printError(ex);
    }
}

cl::Event ActiveContours::copyBufToImg(cl::Buffer& buf, cl::Image2D& img, vector<cl::Event> vecEvPrev, bool bufHasAllBands) {
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

cl::Event ActiveContours::copyTextToBuf(cl::Image2DGL& text, cl::Buffer& buf, vector<cl::Event> vecEvPrev) {
    if (WRITE) {
        cout << "----------- Copying texture to buffer ------------" << endl;
    }
    cl::Event evBufToText;
    try {
        cl::CommandQueue* queue = clMan.getQueue();
        cl::Program* program = clMan.getProgram();

        cl::Kernel kernelBufToText(*program, (char*) "textToBuf");
        kernelBufToText.setArg(0, text);
        kernelBufToText.setArg(1, buf);
        kernelBufToText.setArg(2, 255); //Multiply each value by 255 TODO (put as parameter)

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
 * This method obtains the average pixel value of the input image for the pixels
 * 'outside' and 'inside' the mask.
 * @param img_mask cl::Image2D& Reference to the mask image
 * @param img_in   cl::Image2D& Reference to the input image
 * @param vecPrevEvents vector<cl::Event> List of events that need to be finished before start
 * @return cl::Event Last event required to compute the averages.
 */
cl::Event ActiveContours::compAvgInAndOut(cl::Image2D& img_phi, cl::Image2D& img_in,
        vector<cl::Event> vecPrevEvents, bool useAllBands) {

    if (WRITE) {
        cout << "Computing average pixel value for Phi > 0 and Phi < 0 ......" << endl;
    }

    cl::Event evStep2Avg;
    try {
        int initSize = tot_grps_x * tot_grps_y * 2; //


        cl::Context* context = clMan.getContext();
        cl::CommandQueue* queue = clMan.getQueue();
        cl::Program* program = clMan.getProgram();

        // This array will be used to count the pixel of each local sum
        int* arr_avg_count = new int[initSize];
        cl::Buffer buf_avg_count = cl::Buffer(*context, CL_MEM_READ_WRITE,
                (size_t) initSize * sizeof (int), NULL, &err);


        cl::Kernel kernelAVGcolor(*program, (char*) "Step1AvgInOut");
        kernelAVGcolor.setArg(0, img_phi);
        kernelAVGcolor.setArg(1, img_in);
        kernelAVGcolor.setArg(2, buf_avg_in_out);
        kernelAVGcolor.setArg(3, buf_avg_count);
        kernelAVGcolor.setArg(4, (int)useAllBands);

        cl::Event evStep1Avg;
        queue->enqueueNDRangeKernel(
                kernelAVGcolor,
                cl::NullRange,
                cl::NDRange((size_t) width, (size_t) height),
                cl::NDRange((size_t) grp_size_x, (size_t) grp_size_y),
                &vecPrevEvents,
                &evStep1Avg);


        vector<cl::Event> vecStep1Avg;
        vecStep1Avg.push_back(evStep1Avg);

        int curr_size = tot_grps_x * tot_grps_y * 2;

        int loc_tot_grps_x = 0;
        int loc_tot_grps_y = 0;

        int loc_grp_size_x = 0;
        int loc_grp_size_y = 0;

        cl::Kernel kernelAVGcolorStep2(*program, (char*) "Step2AvgInOut");
        kernelAVGcolorStep2.setArg(0, buf_avg_in_out);
        kernelAVGcolorStep2.setArg(1, buf_avg_count);

        //Set the events for the second kernel
        vector<cl::Event> vecStep2Avg;
        vecStep2Avg.push_back(evStep1Avg);

        do {
            // Updating the dimensions
            clMan.getGroupSize(max_warp_size, curr_size, 1,
                    loc_grp_size_x, loc_grp_size_y, loc_tot_grps_x, loc_tot_grps_y, false);

            if (WRITE) {
                cout << endl << "----------- Current Size: " << curr_size << "------------" << endl;
                cl::Event evReadAvg;
                cl::Event evReadCount;

                res = queue->enqueueReadBuffer(buf_avg_in_out,
                        CL_TRUE, (size_t) 0, (size_t) (sizeof (float) *curr_size),
                        (void*) arr_avgInOutByGrp, &vecStep2Avg, &evReadAvg);

                res = queue->enqueueReadBuffer(buf_avg_count,
                        CL_TRUE, (size_t) 0, (size_t) (sizeof (float) *curr_size),
                        (void*) arr_avg_count, &vecStep2Avg, &evReadCount);

                //				vecReadBuf.push_back(evReadAvg);
                //				vecReadBuf.push_back(evReadCount);
                //                queue->enqueueWaitForEvents(vecReadBuf);

                float countIn = 0;
                float countOut = 0;

                float avgIn = 0;
                float avgOut = 0;

                for (int j = 0; j < curr_size - 1; j += 2) {
                    //					cout << arr_avgInOutByGrp[j] << "\t" << arr_avgInOutByGrp[j+1] << endl;
                    if (arr_avg_count[j] != 0) {
                        avgIn += arr_avgInOutByGrp[j];
                        countIn += arr_avg_count[j];
                    }
                    if (arr_avg_count[j + 1] != 0) {
                        avgOut += arr_avgInOutByGrp[j + 1];
                        countOut += arr_avg_count[j + 1];
                    }
                }
                cout << "Sum in =" << avgIn << " Sum out=" << avgOut << endl;
                cout << "Count in =" << countIn << " Count out=" << countOut << endl;

                avgIn = avgIn / countIn;
                avgOut = avgOut / countOut;
                cout << "Avg in =" << avgIn << " out=" << avgOut << endl;
            }

            queue->enqueueNDRangeKernel(
                    kernelAVGcolorStep2,
                    cl::NullRange,
                    cl::NDRange((size_t) curr_size, (size_t) 1),
                    cl::NDRange((size_t) loc_grp_size_x, (size_t) loc_grp_size_y),
                    &vecStep2Avg,
                    &evStep2Avg);

            vecStep2Avg.push_back(evStep2Avg);
            queue->enqueueWaitForEvents(vecStep2Avg);
            vecStep2Avg.clear();

            curr_size = loc_tot_grps_x * loc_tot_grps_y * 2;
        } while (loc_tot_grps_x * loc_tot_grps_y > 1);


    } catch (cl::Error ex) {
        cout << "Error at compAvgInandOut" << endl;
        clMan.printError(ex);
    }
    return evStep2Avg;
}

cl::Event ActiveContours::smoothPhi(vector<cl::Event> vecEvPrev, float alpha) {
    if (WRITE) {
        cout << "----------- Smoothing new phi ------------" << endl;
    }
    cl::Event evSmooth;
    try {
        cl::CommandQueue* queue = clMan.getQueue();
        cl::Program* program = clMan.getProgram();

        cl::Kernel kernelSmoothPhi(*program, (char*) "smoothPhi");
        kernelSmoothPhi.setArg(0, img_newphi);
        kernelSmoothPhi.setArg(1, img_phi);
        kernelSmoothPhi.setArg(2, alpha);

        // Do the work
        queue->enqueueNDRangeKernel(
                kernelSmoothPhi,
                cl::NullRange,
                cl::NDRange((size_t) width, (size_t) height),
                cl::NDRange((size_t) grp_size_x, (size_t) grp_size_y),
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
    cl::Event evNewPhi;
    try {
        cl::Context* context = clMan.getContext();
        cl::CommandQueue* queue = clMan.getQueue();
        cl::Program* program = clMan.getProgram();

        cl::Sampler sampler(*context, CL_FALSE,
                CL_ADDRESS_REPEAT, CL_FILTER_NEAREST, &err);

        cl::Kernel kernelDphiDt(*program, (char*) "newphi");
        kernelDphiDt.setArg(0, img_phi);
        kernelDphiDt.setArg(1, img_dphidt);
        kernelDphiDt.setArg(2, img_newphi);
        kernelDphiDt.setArg(3, sampler);
        kernelDphiDt.setArg(4, buf_max_dphidt);

        // Do the work
        queue->enqueueNDRangeKernel(
                kernelDphiDt,
                cl::NullRange,
                cl::NDRange((size_t) width, (size_t) height),
                cl::NDRange((size_t) grp_size_x, (size_t) grp_size_y),
                &vecEvPrev,
                &evNewPhi);

    } catch (cl::Error ex) {
        clMan.printError(ex);
    }
    return evNewPhi;
}

cl::Event ActiveContours::compDphiDt(vector<cl::Event> vecEvPrev) {
    cl::Event evCompDphiDt;
    try {
        cl::Context* context = clMan.getContext();
        cl::CommandQueue* queue = clMan.getQueue();
        cl::Program* program = clMan.getProgram();

        cl::Sampler sampler(*context, CL_FALSE,
                CL_ADDRESS_REPEAT, CL_FILTER_NEAREST, &err);

        cl::Kernel kernelDphiDt(*program, (char*) "dphidt");
        kernelDphiDt.setArg(0, img_phi);
        kernelDphiDt.setArg(1, img_curv_F);
        kernelDphiDt.setArg(2, img_dphidt);
        kernelDphiDt.setArg(3, buf_F);
        kernelDphiDt.setArg(4, sampler);
        kernelDphiDt.setArg(5, alpha);

        // Do the work
        queue->enqueueNDRangeKernel(
                kernelDphiDt,
                cl::NullRange,
                cl::NDRange((size_t) width, (size_t) height),
                cl::NDRange((size_t) grp_size_x, (size_t) grp_size_y),
                &vecEvPrev,
                &evCompDphiDt);

    } catch (cl::Error ex) {
        clMan.printError(ex);
    }
    return evCompDphiDt;
}

cl::Event ActiveContours::compCurvAndF(vector<cl::Event> vecEvPrev, bool useAllBands) {

    cl::Event evCurvF;
    if (WRITE) {
        cout << endl << " ----------------- Computing Curvature and F ---------" << endl;
    }
    try {
        cl::Context* context = clMan.getContext();
        cl::CommandQueue* queue = clMan.getQueue();
        cl::Program* program = clMan.getProgram();

        cl::Sampler sampler(*context, CL_FALSE,
                CL_ADDRESS_REPEAT, CL_FILTER_NEAREST, &err);

        cl::Kernel kernelCurvAndF(*program, (char*) "CurvatureAndF");
        kernelCurvAndF.setArg(0, img_phi);
        kernelCurvAndF.setArg(1, img_in);
        kernelCurvAndF.setArg(2, img_curv_F);
        kernelCurvAndF.setArg(3, sampler);
        kernelCurvAndF.setArg(4, buf_avg_in_out); // TODO SEND just a subbuffer
        kernelCurvAndF.setArg(5, (int)useAllBands); // TODO SEND just a subbuffer


        queue->enqueueNDRangeKernel(
                kernelCurvAndF,
                cl::NullRange,
                cl::NDRange((size_t) width, (size_t) height),
                cl::NDRange((size_t) grp_size_x, (size_t) grp_size_y),
                &vecEvPrev,
                &evCurvF);

    } catch (cl::Error ex) {
        clMan.printError(ex);
    }
    return evCurvF;
}

/**
 * Computes the maximum value of a float array. 
 * 
 * @param buf_in
 * @param curr_size
 * @param absVal
 * @return 
 */
cl::Event ActiveContours::compArrMaxReduction(cl::Buffer& buf_in, int curr_size, bool absVal,
        vector<cl::Event> vecPrev) {

    //Block size is the number of cells that each tread is going to check 
    int maxBlockSize = 128; //TODO set this as a global variable (think best option)

    int block = min(curr_size, maxBlockSize);

    while (curr_size % block != 0) {
        block--;
    }

    if (WRITE) {
        cout << "----- Computing maximum of an array ------------" << endl;
        //        cout << " Final block size: " << block << endl;
    }

    int loc_tot_grps_x = 0;
    int loc_tot_grps_y = 0;

    int loc_grp_size_x = 0;
    int loc_grp_size_y = 0;

    cl::CommandQueue* queue = clMan.getQueue();
    cl::Program* program = clMan.getProgram();

    cl::Kernel kernelArrMax(*program, (char*) "ArrMaxFloat");
    kernelArrMax.setArg(0, buf_in);
    kernelArrMax.setArg(1, block);
    kernelArrMax.setArg(2, curr_size);
    kernelArrMax.setArg(3, absVal ? 1 : 0);

    //Set the events for the second kernel

    cl::Event evMaxEven;
    cl::Event evMaxOdd;
    cl::Event evLast;

    vector<cl::Event> vecEvMaxEven = vecPrev; // Used for even iterations
    vector<cl::Event> vecEvMaxOdd; // Used for odd iterations
    vector<cl::Event> vecLast; // Always used

    int iter = 0;

    int num_threads = curr_size / block; //Number of threads to use

    while (curr_size > 1) {

        clMan.getGroupSize(max_warp_size, num_threads, 1,
                loc_grp_size_x, loc_grp_size_y, loc_tot_grps_x, loc_tot_grps_y, false);

        if (iter % 2 == 0) {//This will be run at the first iteration
            queue->enqueueNDRangeKernel(
                    kernelArrMax,
                    cl::NullRange,
                    cl::NDRange((size_t) num_threads, (size_t) 1),
                    cl::NDRange((size_t) loc_grp_size_x, (size_t) loc_grp_size_y),
                    &vecEvMaxEven,
                    &evMaxEven);

            evLast = evMaxEven;
            vecEvMaxOdd.clear();
            vecEvMaxOdd.push_back(evMaxEven);

            vecLast.clear();
            vecLast.push_back(evMaxEven);
        } else {
            queue->enqueueNDRangeKernel(
                    kernelArrMax,
                    cl::NullRange,
                    cl::NDRange((size_t) num_threads, (size_t) 1),
                    cl::NDRange((size_t) loc_grp_size_x, (size_t) loc_grp_size_y),
                    &vecEvMaxOdd,
                    &evMaxOdd);

            evLast = evMaxOdd;
            vecEvMaxEven.clear();
            vecEvMaxEven.push_back(evMaxOdd);

            vecLast.clear();
            vecLast.push_back(evMaxOdd);
        }

        // TODO verify that this is always correct ( the new size is always
        // the previous number of threads
        curr_size = num_threads;
        block = min(curr_size, maxBlockSize);
        num_threads = ceil(curr_size / block); //Number of threads to use

        if (WRITE) {
            float* result = new float[curr_size];
            res = queue->enqueueReadBuffer(buf_in,
                    CL_TRUE, (size_t) 0, (size_t) (sizeof (float) *curr_size),
                    (void*) result, &vecLast, 0);

            float max_value = 0;
            cout << "---- Current size: " << curr_size << endl;
            for (int i = 0; i < curr_size; i++) {
                cout << result[i] << endl;
                if (result[i] > max_value) {
                    max_value = result[i];
                }
            }
            cout << "Max value: " << max_value << endl;
            delete[] result;
        }
        iter++;
    }

    return evLast;
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
cl::Event ActiveContours::compImgMaxReduction(cl::Image2D& img, vector<cl::Event> vecEvPrev,
        cl::Buffer& buf_out, int layer, bool absVal) {

    if (WRITE) {
        cout << endl << "--------- Computing maximum value -------------" << endl;
    }

    cl::Event evRedStep1;
    cl::Event evRedStep2;

    int local_grp_size_x;
    int local_grp_size_y;
    int local_tot_grps_x;
    int local_tot_grps_y;

    // These part creates groups sizes that will use all the width of the image
    CLManager::getGroupSize(max_warp_size, width, 1, local_grp_size_x, local_grp_size_y, local_tot_grps_x, local_tot_grps_y, WRITE);

    int size_array = (int) local_tot_grps_x;

    try {
        cl::Context* context = clMan.getContext();
        cl::CommandQueue* queue = clMan.getQueue();
        cl::Program* program = clMan.getProgram();

        cl::Kernel kernelMaxValue(*program, (char*) "ImgColMax");

        cl::Sampler sampler(*context, CL_FALSE,
                CL_ADDRESS_REPEAT, CL_FILTER_NEAREST, &err);

        kernelMaxValue.setArg(0, img);
        kernelMaxValue.setArg(1, sizeof (float) * local_grp_size_x, NULL);
        kernelMaxValue.setArg(2, buf_out);
        kernelMaxValue.setArg(3, sampler);
        kernelMaxValue.setArg(4, layer); //Layer 1 'Red', 2 'Green', 3 'Blue', etc.
        kernelMaxValue.setArg(5, absVal ? 1 : 0); // Don't use absolute value

        // Do the work
        queue->enqueueNDRangeKernel(
                kernelMaxValue,
                cl::NullRange,
                cl::NDRange((size_t) width, (size_t) 1),
                cl::NDRange((size_t) local_grp_size_x, (size_t) 1),
                &vecEvPrev,
                &evRedStep1);

        vector<cl::Event> vecEvRedStep1;
        vecEvRedStep1.push_back(evRedStep1);

        if (WRITE) {
            float* result = new float[size_array];
            res = queue->enqueueReadBuffer(buf_out,
                    CL_TRUE, (size_t) 0, (size_t) (sizeof (float) *size_array),
                    (void*) result, &vecEvRedStep1, 0);

            float max_value = 0;
            cout << "---- Current size: " << size_array << endl;
            for (int i = 0; i < size_array; i++) {
                if (result[i] > max_value) {
                    max_value = result[i];
                }
            }
            cout << "Max value: " << max_value << endl;
            delete[] result;
        }

        if (size_array > 1) {//If there is only one element, then we already have the maximum
            evRedStep2 = compArrMaxReduction(buf_out, size_array, absVal, vecEvRedStep1);
        } else {
            evRedStep2 = evRedStep1; //If we didn't make step 2, then the event to wait is event 1
        }

    } catch (cl::Error ex) {
        clMan.printError(ex);
    }

    return evRedStep2;
}

void ActiveContours::createRGBAMask(int width, int height, int xstart, int xend,
        int ystart, int yend) {

    arr_buf_mask = new float[width * height];

    //Update local width and height of the image
    width = width;
    height = height;

    int size = width * height;
    int indx = 0;

    cout << "--------------------- Creating mask (" << width << "," << height << ") ----------" << endl;
    dout << "xmin: " << xstart << " xmax: " << xend << endl;
    dout << "ymin: " << ystart << " ymax: " << yend << endl;

    //Initialize to 0
    for (int i = 0; i < size; i++) {
        arr_buf_mask[i] = 0; // Red value
    }

    //Set the internal mask to 255
    for (int i = ystart; i < yend; i++) {
        for (int j = xstart; j < xend; j++) {
            indx = ImageManager::indxFromCoord(width, i, j, 1);
            arr_buf_mask[indx] = 255; //R
        }
    }

   //if (PRINT_IMG_VAL) {
        //cout << "Mask (just after creating it)" << endl;;
        //ImageManager::printImage(width, height, arr_buf_mask, 1);
    //}
}

/*
   float* ActiveContours::createRGBAMask(int width, int height, int xstart, int xend,
   int ystart, int yend) {
   int dim = 4; // RGBA dimension
   float* mask = new float[width * height * dim];
   int size = width * height*dim;
   int indx = 0;

   cout << " Creating mask...." << endl;
   cout << "xmin: " << xstart << " xmax: " << xend << endl;
   cout << "ymin: " << ystart << " ymin: " << yend << endl;

   for (int i = 0; i < size - dim; i += dim) {
   mask[i] = 0; // Red value
   mask[i + 1] = 0; //Green value
   mask[i + 2] = 0; // Blue value
   mask[i + 3] = 255; // Alpha value
   }

   for (int i = ystart; i < yend; i++) {
   for (int j = xstart; j < xend; j++) {
   indx = ImageManager::indxFromCoord(width, i, j, dim);
   mask[indx] = 100;
   }
   }

   return mask;
   }
 */
