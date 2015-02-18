
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

#define PI 3.14159265359

FPSMovement::FPSMovement(float fzNear, float fzFar, float FOV) : CameraMovement(fzNear,fzFar, FOV)
{
	translating = false;
	rotating = false;
    dout << "Inside FPSMovement " << endl;
	prevRot = glm::mat4(1.0f);
}

FPSMovement::FPSMovement(const FPSMovement& orig) :CameraMovement(orig)
{
    dout << "Copy const FPSMov " << endl;
}

FPSMovement::~FPSMovement() 
{
}

void FPSMovement::rotateModel(float radX, float radY, float radZ){
	glm::vec4 new_x_axis = prevRot*(glm::vec4(0.0f,1.0f,0.0f,0.0f));
	glm::vec4 new_y_axis = prevRot*(glm::vec4(1.0f,0.0f,0.0f,0.0f));
	glm::vec4 new_z_axis = prevRot*(glm::vec4(0.0f,0.0f,1.0f,0.0f));
    
    //        cout << "Rotating ..." << endl; 
    //		printf("%2.2f \t %2.2f \t %2.2f \t %2.2f \n", new_x_axis[0], new_x_axis[1],
    //				new_x_axis[2], new_x_axis[3]);
    
    modelMatrix = glm::rotate( modelMatrix, radX, glm::vec3(new_x_axis));
    modelMatrix = glm::rotate( modelMatrix, radY, glm::vec3(new_y_axis));
    modelMatrix = glm::rotate( modelMatrix, radZ, glm::vec3(new_z_axis));
    
	//This matrix is used to modify the axis of rotation
	prevRot = glm::rotate( prevRot, -radX, glm::vec3(0.0f,1.0f,0.0f));
	prevRot = glm::rotate( prevRot, -radY, glm::vec3(1.0f,0.0f,0.0f));
    prevRot = glm::rotate( prevRot, -radZ, glm::vec3(0.0f,0.0f,1.0f));
}

void FPSMovement::mouseMoveEvent(QMouseEvent *event)
{
    if(rotating){
        int newX = event->x();
        int newY = event->y();
        float movInX = initX - newX;
        float movInY = initY - newY;
        
        float reduceSpeedBy = 2;
		//make the movement slower
		movInX = movInX/reduceSpeedBy;
		movInY = movInY/reduceSpeedBy;

        float half_width = (float) win_width/2;
        float farFromCenter = (float)(newX - half_width)/half_width;
        farFromCenter = farFromCenter/reduceSpeedBy;
        
		rotateModel(-movInX, -movInY, farFromCenter*movInY);
        
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

        switch (key) {
            case '1'://Reset view
                modelMatrix = glm::mat4(1.0f);
                prevRot = glm::mat4(1.0f);
                break;
            case '2'://set view to look from above
                modelMatrix = glm::mat4(1.0f);
                prevRot = glm::mat4(1.0f);
                rotateModel(90, 0.0f, 0.0f);
                break;
            case '3'://set view to look from above
                modelMatrix = glm::mat4(1.0f);
                prevRot = glm::mat4(1.0f);
                rotateModel(0.0f, 90, 0.0f);
                break;
        }
    }
}

/* Catches all release event keys
*/
void FPSMovement::keyReleaseEvent(QKeyEvent* event)
{
    unsigned char key = event->key();

    if(!event->isAutoRepeat() ) {    
        cout << "Keyboard released on FPSMovement (not autorepeat)" << endl;

        switch (key) {
            break;
        }
    }
}
void FPSMovement::printGLMmatrix(glm::mat4 matrix)
{
    printf("%2.2f \t %2.2f \t %2.2f \t %2.2f \n", matrix[0].x, matrix[0].y, matrix[0].z, matrix[0].w);
    printf("%2.2f \t %2.2f \t %2.2f \t %2.2f \n", matrix[1].x, matrix[1].y, matrix[1].z, matrix[1].w);
    printf("%2.2f \t %2.2f \t %2.2f \t %2.2f \n", matrix[2].x, matrix[2].y, matrix[2].z, matrix[2].w);
    printf("%2.2f \t %2.2f \t %2.2f \t %2.2f \n", matrix[3].x, matrix[3].y, matrix[3].z, matrix[3].w);
}

