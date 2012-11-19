
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
    camMatrix[0].x = fFrustumScale / (w/ (float)h);
    camMatrix[1].y = fFrustumScale;
}

CameraMovement::CameraMovement(float fzNear, float fzFar, float FOV)
{
    dout << "Inside CameraMovement" << endl;

    //Defines the initial perspective matrix
    camMatrix = glm::mat4(1.0f);

    fFrustumScale = this->CalcFrustrumScale(FOV);

    camMatrix[0].x = fFrustumScale;
    camMatrix[1].y = fFrustumScale;
    camMatrix[2].z = (fzFar + fzNear) / (fzNear - fzFar);
    camMatrix[2].w = -1.0f;
    camMatrix[3].z = (2 * fzFar * fzNear) / (fzNear - fzFar);
}


glm::mat4 CameraMovement::getCameraMatrix(){
    return camMatrix;
}
