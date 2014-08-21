
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

void FPSMovement::mouseMoveEvent(QMouseEvent *event)
{
    if(rotating){
        int newX = event->x();
        int newY = event->y();
        float movInX = initX - newX;
        float movInY = initY - newY;

		//make the movement slower
		movInX = movInX/2;
		movInY = movInY/2;

//        cout << "Rotating ..." << endl; 
        modelMatrix = glm::rotate( modelMatrix, (float)-movInX, glm::vec3(0.0f,1.0f,0.0f));
        modelMatrix = glm::rotate( modelMatrix, (float)-movInY, glm::vec3(1.0f,0.0f,0.0f));

        initX = newX;
        initY = newY;
    }
    if(translating){
        int newX = event->x();
        int newY = event->y();
        int movInX = initX - newX;
        int movInY = initY - newY;

//        cout << "Translating ..." << endl; 
        viewMatrix = glm::translate( viewMatrix,
                glm::vec3(-movInX*movementSpeed,movInY*movementSpeed,0));

        initX = newX;
        initY = newY;
    }
}

void FPSMovement::mousePressEvent(QMouseEvent *event)
{
    cout << "Inside press event of FPSMovement" << endl;
    int button = event->button();
    switch(button){
        case PRIMARY:
            if(event->modifiers().testFlag(Qt::ControlModifier)){
                cout << "Will Translate... " << endl;
                translating = true;
            }else{
                cout << "Will Rotate ... " << endl;
                rotating = true;
            }
            break;
    }

    initX = event->x();
    initY = event->y();
}

void FPSMovement::wheelEvent(QWheelEvent* event){
	int movement =  event->delta();

	viewMatrix = glm::translate( viewMatrix,
			glm::vec3(0,0,movement*movementSpeed));
	
}

void FPSMovement::mouseReleaseEvent(QMouseEvent *event)
{ 
    cout << "Inside Release Event of FPSMovement" << endl;
    int button = event->button();

    switch(button){
        case PRIMARY:
            translating = false;
            rotating = false;
            break;
    }

}

/* Catches all pressed event keys
*/
void FPSMovement::keyPressEvent(QKeyEvent* event)
{
    unsigned char key = event->key();

    if(!event->isAutoRepeat() ) {    
        cout << "Keyboard pressed on FPSMovement (not autorepeat)" << endl;
        glm::mat4 tempMat(1.0f);

        switch (key) {
            break;
        }
        projMatrix = projMatrix*tempMat;
    }
}

/* Catches all release event keys
*/
void FPSMovement::keyReleaseEvent(QKeyEvent* event)
{
    unsigned char key = event->key();

    if(!event->isAutoRepeat() ) {    
        cout << "Keyboard released on FPSMovement (not autorepeat)" << endl;
        glm::mat4 tempMat(1.0f);

        switch (key) {
            break;
        }
        projMatrix = projMatrix*tempMat;
    }
}
