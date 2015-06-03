/* 
 * File:   Tools.cpp
 * Author: olmozavala
 * 
 * Created on August 26, 2014, 1:55 PM
 */

#include "Tools.h"
#include <stdlib.h>
#include <iostream>
#include "debug.h"

using namespace std;
Tools::Tools() {
}

Tools::Tools(const Tools& orig) {
}

Tools::~Tools() {
}

void Tools::validateGLlocations(GLuint glLoc, char* name){
	if (glLoc == -1){
		eout << "ERROR: Can't locate uniform: " << name << endl;
	}
}