PROJECT(grabcut)
 
FIND_PACKAGE( OpenCV REQUIRED )   

INCLUDE_DIRECTORIES(.)
ADD_EXECUTABLE(grabcut grabcut.cpp   )
TARGET_LINK_LIBRARIES (grabcut ${OpenCV_LIBS} )   #这两行的次序也不能变!

INCLUDE_DIRECTORIES(.)
ADD_EXECUTABLE(surf histogrammatch.cpp   )
TARGET_LINK_LIBRARIES (surf ${OpenCV_LIBS} )   #这两行的次序也不能变!
