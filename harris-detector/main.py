import cv2
import numpy as np

WITH_NMS = True         #是否非极大值抑制
threshold = 0.1         #设定阈值  

def getM(img):
    h, w = img.shape[:2]
    Ix = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
    Iy = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)

    #2、计算Ix^2,Iy^2,Ix*Iy
    m = np.zeros((h,w,3),dtype=np.float32)
    m[:,:,0] = Ix**2
    m[:,:,1] = Iy**2
    m[:,:,2] = Ix*Iy

    ksize = (3,3)
    m[:,:,0] = cv2.GaussianBlur(m[:,:,0],ksize=ksize,sigmaX=2)
    m[:,:,1] = cv2.GaussianBlur(m[:,:,1],ksize=ksize,sigmaX=2)
    m[:,:,2] = cv2.GaussianBlur(m[:,:,2],ksize=ksize,sigmaX=2)
    m = [np.array([[m[i,j,0],m[i,j,2]],[m[i,j,2],m[i,j,1]]]) for i in range(h) for j in range(w)]
    return m

def calcR(M, h, w):
    k = 0.04
    #4、计算局部特征结果矩阵M的特征值和响应函数R(i,j)=det(M)-k(trace(M))^2  0.04<=k<=0.06
    D,T = list(map(np.linalg.det,M)),list(map(np.trace,M))
    R = np.array([d-k*t**2 for d,t in zip(D,T)])

    normR = (R / np.linalg.norm(R))*255
    R_max = np.max(normR)
    print(R_max)
    print(np.min(normR))
    R = normR.reshape(h,w)
    cv2.imshow("R", R)
    corner = np.zeros_like(R,dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if WITH_NMS:
                #除了进行进行阈值检测 还对3x3邻域内非极大值进行抑制(导致角点很小，会看不清)
                if R[i,j] > R_max*threshold and R[i,j] == np.max(R[max(0,i-1):min(i+2,h-1),max(0,j-1):min(j+2,w-1)]):
                    corner[i,j] = 255
            else:
                #只进行阈值检测
                if R[i,j] > R_max*threshold :
                    corner[i,j] = 255
    return corner

def markCorner(img, R):
    print("R shape is:", R.shape)
    h, w = img.shape[0:2]
    print("img shape is :", img.shape)
    h, w = img.shape[:2]
    Rmax = R.max()
    color = (0, 0, 255)
    for i in range(h):
        for j in range(w):
            if R[i][j] > threshold * Rmax:
                cv2.circle(img, (j, i), 4, color, thickness=-1)
    # img[R>0.01*R.max()] = [0,0,255]
    return

def lambdaPic(M, h, w):
    tmp = np.zeros((2,2), dtype=np.float32)
    lambdaMinMap = np.zeros((h, w), dtype=np.float32)
    lambdaMaxMap = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            step = i*w+j
            tmp = M[step]
            eigValue, vects = np.linalg.eig(tmp)
            lambdaMinMap[i][j] = min(eigValue[0], eigValue[1])
            lambdaMaxMap[i][j] = max(eigValue[0], eigValue[1])
    normMax = (lambdaMaxMap / np.linalg.norm(lambdaMaxMap))*255
    normMin = (lambdaMinMap / np.linalg.norm(lambdaMinMap))*255
    cv2.imshow("lambdaMax", normMax)
    cv2.imshow("lambdaMin", normMin)

if __name__ == "__main__":
    img = cv2.imread("horse.jpg")
    # paint eigMax and eigMin for gray image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    M = getM(gray)
    lambdaPic(M, gray.shape[0], gray.shape[1])
    R = calcR(M, gray.shape[0], gray.shape[1])
    markCorner(img, R)
    cv2.imshow("harris corner", img)
    cv2.waitKey(0)