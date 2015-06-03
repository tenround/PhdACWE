/* 
 * File:   Tools.h
 * Author: olmozavala
 *
 * Created on August 26, 2014, 1:55 PM
 */

#include <GL/gl.h>

#ifndef TOOLS_H
#define	TOOLS_H

class Tools {
public:
    Tools();
    Tools(const Tools& orig);
    virtual ~Tools();

	static void validateGLlocations(GLuint glLoc, char* name);
private:

};

#endif	/* TOOLS_H */

