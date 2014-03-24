
/* 
 * File:   FPSMovement.cpp
 * Author: Olmo Zavala Romero 
 * 
 */

#include "FPSMovement.h"
#include <stdlib.h>
#include <iostream>
#include "debug.h"

using namespace std;

FPSMovement::FPSMovement(float fzNear, float fzFar, float FOV) : CameraMovement(fzNear,fzFar, FOV)
{
    dout << "Inside FPSMovement " << endl;
}

FPSMovement::FPSMovement(const FPSMovement& orig) :CameraMovement(orig)
{
    dout << "Copy const FPSMov " << endl;
}

FPSMovement::~FPSMovement() 
{
}


void FPSMovement::MMotion(int x, int y)
{
    cout << x << " - " << y << endl;
}

void FPSMovement::MButton(int button, int state, int x, int y)
{
    switch(button){
        case GLUT_LEFT_BUTTON:
            break;
        case GLUT_RIGHT_BUTTON:
            break;
    }

}

void FPSMovement::Keyboard(unsigned char key, int x, int y)
{
    dout << "Keyboard on FPSMovement" << endl;
    glm::mat4 tempMat(1.0f);
    switch (key)
    {

        case 97:// Case 'A' go left 
        case 65:
            tempMat[3].x = tempMat[3].x + .1;
            dout << "A" << endl;
            break;

        case 100:// Case 'D' go right 
        case 68:
            //dout << "D" << endl;
            tempMat[3].x = tempMat[3].x - .1;
            break;

        case 119:// Case 'W' go forward
        case 87:
            tempMat[3].z = tempMat[3].z + .1;
            //dout << "W" << endl;
            break;

        case 83:// Case 'S' go back 
        case 115:
            tempMat[3].z = tempMat[3].z - .1;
            //dout << "S" << endl;
            break;
            
        case 27: 
            glutLeaveMainLoop();
    }

    camMatrix = camMatrix*tempMat;
}
