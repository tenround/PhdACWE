
/* 
 * File:   CameraMovement.h
 * Author: Olmo Zavala Romero
 *
 */

#ifndef CAMERAMOV_H
#define	CAMERAMOV_H

#include <glm/glm.hpp>
#include <GL/freeglut.h>


class CameraMovement {
public:
	CameraMovement(float fzNear, float fzFar, float FOV);
    virtual void MMotion(int x, int y) = 0;
    virtual void MButton(int button, int state, int x, int y) = 0;
    virtual void Keyboard(unsigned char key, int x, int y) = 0; 

    void Reshape(int w, int h);
    glm::mat4 getCameraMatrix();
    

protected:
    glm::mat4 camMatrix;//Original perspective matrix
    float fFrustumScale;

private:
    float CalcFrustrumScale(float FOVdeg);

};

#endif	/* CameraMovement_H */
