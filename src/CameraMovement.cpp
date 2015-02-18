
/* 
 * File:   CameraMovement.cpp
 * Author: Olmo Zavala Romero 
 * 
 */

#include "CameraMovement.h"
#include <stdlib.h>
#include <iostream>
#include <math.h>

#include "debug.h"

#define PI 3.14159

using namespace std;

float CameraMovement::CalcFrustrumScale(float FOVdeg)
{
    float degToRad = PI * 2.0f/360.0f;
    float fovRad = FOVdeg * degToRad;
    return 1.0f/ tan(fovRad/2.0f); 
}

void CameraMovement::Reshape(int w, int h)
{
    win_width = w;
    win_height = h;
    cout << "Modifying projection matrix... " << endl;
//    projMatrix[0].x = fFrustumScale / (w/ (float)h);
//    projMatrix[1].y = fFrustumScale;
	cout << "Width: " << w << "   Height: " << h << endl;
    projMatrix = glm::perspective(FOV,  (float)w/(float)h, fzNear, fzFar);
}

/**
 * Initializes the parameter of the perspective and view matrix. 
 * 
 * IMPORTANT. By default (if we leave the prespective and view matrix as identity
 * matrices) OpenGL will position the camera at (0,0,1), looking at (0,0,0)
 * and it will be able to see from (-1,1) in X and Y in orthographic projection
 * 
 * @param fzNear Closest distance from the camera that we are able to see
 * @param fzFar  Farthest distance from camera that we are able to see
 * @param FOV
 */
CameraMovement::CameraMovement(float fzNear, float fzFar, float FOV)
{
	this->fzNear = fzNear;
	this->fzFar = fzFar;
	this->FOV = FOV;

    dout << "Inside CameraMovement" << endl;

    //Defines the initial perspective matrix
    projMatrix = glm::mat4(1.0f);
    modelMatrix = glm::mat4(1.0f);
    viewMatrix = glm::mat4(1.0f);

    //Giving initial values 
    win_height = 0;
    win_width = 0;
	/*
    fFrustumScale = this->CalcFrustrumScale(FOV);

    projMatrix[0].x = fFrustumScale;
    projMatrix[1].y = fFrustumScale;
    projMatrix[2].z = (fzFar + fzNear) / (fzNear - fzFar);
    projMatrix[2].w = -1.0f;
    projMatrix[3].z = (2 * fzFar * fzNear) / (fzNear - fzFar);
	 */

	//From whatever the position of the camera is (viewMatrix) the 
	// camara will see from .1 infront to 100
    projMatrix = glm::perspective(FOV, 4.0f / 3.0f, fzNear, fzFar);
    // Camera matrix
    viewMatrix = glm::lookAt(
        glm::vec3(0,0,3), // Camera is at (4,3,3), in World Space
        glm::vec3(0,0,0), // and looks at the origin
        glm::vec3(0,1,0)  // Head is up (set to 0,-1,0 to look upside-down)
    );
}

glm::mat4 CameraMovement::getCameraMatrix(){
    return projMatrix;
}
glm::mat4 CameraMovement::getViewMatrix(){
    return viewMatrix;
}
glm::mat4 CameraMovement::getModelMatrix(){
    return modelMatrix;
}
glm::mat4 CameraMovement::getProjectionMatrix(){
    return projMatrix;
}

void CameraMovement::setModelMatrix(glm::mat4 newModelMatrix){
    modelMatrix = newModelMatrix;
}
