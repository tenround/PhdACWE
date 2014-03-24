
/* 
 * File:   FPSMovement.h
 * Author: Olmo Zavala Romero
 *
 */

#ifndef FPSMOV_H
#define	FPSMOV_H

#include "CameraMovement.h"
#include <glm/glm.hpp>

class FPSMovement : public CameraMovement{

public:
    FPSMovement(float fzNear, float fzFar, float FOV);
    FPSMovement(const FPSMovement& orig);
    virtual ~FPSMovement();

    void MMotion(int x, int y);
    void MButton(int button, int state, int x, int y);
    void Keyboard(unsigned char key, int x, int y); 
};

#endif	/* FPSMovement_H */
