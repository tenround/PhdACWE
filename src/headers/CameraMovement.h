
/* 
 * File:   CameraMovement.h
 * Author: Olmo Zavala Romero
 *
 */

#ifndef CAMERAMOV_H
#define	CAMERAMOV_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <GL/freeglut.h>

#include <QtGui/QMouseEvent>

class CameraMovement {
public:
	CameraMovement(float fzNear, float fzFar, float FOV);
    virtual void mousePressEvent(QMouseEvent *event) = 0;
    virtual void mouseReleaseEvent(QMouseEvent *event) = 0;
    virtual void mouseMoveEvent(QMouseEvent *event) = 0;
    virtual void wheelEvent(QWheelEvent* event) = 0;
    virtual void keyPressEvent(QKeyEvent* event) = 0;
    virtual void keyReleaseEvent(QKeyEvent* event) = 0;

    void Reshape(int w, int h);
    glm::mat4 getCameraMatrix();//Deprecated
    glm::mat4 getProjectionMatrix();
    glm::mat4 getModelMatrix();
    void setModelMatrix(glm::mat4 newModelMatrix);
    glm::mat4 getViewMatrix();
    

protected:
    glm::mat4 viewMatrix;//Original perspective matrix
    glm::mat4 projMatrix;//Original perspective matrix
    glm::mat4 modelMatrix;//Original perspective matrix
    float fFrustumScale;
	float fzNear;
	float fzFar;
	float FOV;
    int win_width;
    int win_height;

private:
    float CalcFrustrumScale(float FOVdeg);
};

#endif	/* CameraMovement_H */
