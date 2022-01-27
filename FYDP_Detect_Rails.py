import cv2
import numpy as np

def empty(a):
    pass

path = 'data/OneDrive_2021-10-06/All Photos/DJI_0047.JPG'

def equationoflineVector(x1,y1,x2,y2):
    m=(y2-y1)/(x2-x1)
    b=x2-y2/m
    return m,b
    #x=t
    #y=(m(t-b))

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def gauge_measurement(m1,b1,m2,b2):
    y=0
    gauge=0
    while y<100:
        x1=y/m1+b1
        x2=y/m2+b2
        gauge+=x2-x1
        y+=1
    meangauge=abs(gauge/100)
    print(f'Gauge:{meangauge}')

def showline(line, number):
    for rho, theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.line(img, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
        print(f'line {number}: x1:{x1}, y1:{y1}, x2:{x2}, y2:{y2}')
        m1, b1 = equationofline(x1, y1, x2, y2)
        return m1,b1

def linedetection(lines,cdst):
    n2 = 0
    for n1 in range(0, len(lines)):
        for rho, theta in lines[n1]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            if n1 == 0:
                strong_lines[n2] = lines[n1]
                cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.line(img, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
                print(f'line {n2+1}: x1:{x1}, y1:{y1}, x2:{x2}, y2:{y2}')
                #m1, b1 = equationofline(x1, y1, x2, y2)
                n2 = n2 + 1
                pt1og=pt1
                pt2og=pt2
                theta1=theta
            else:
                if rho < 0:
                    rho *= 1
                    #theta -= np.pi
                closeness_rho = np.isclose(rho, strong_lines[0:n2, 0, 0], atol=50)
                closeness = np.all([closeness_rho], axis=0)
                if not any(closeness) and np.allclose(abs(theta),abs(theta1),rtol=0.000001)==True and intersect(pt1og,pt2og,pt1,pt2)==False: #takes away all lines that are in proximity and only keeps other lines with similar theta
                    strong_lines[n2] = lines[n1]
                    cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
                    cv2.line(img, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
                    #m2, b2 = equationofline(x1, y1, x2, y2)
                    print(f'line {n2+1}: x1:{x1}, y1:{y1}, x2:{x2}, y2:{y2}')
                    n2 = n2 + 1

    #gauge_measurement(m1,b1,m2,b2)

while True:
    img = cv2.imread(path)
    scale_percent = 20  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img=cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    #img=cv2.resize(img,(1280,720))
    dimensions = img.shape
    print(dimensions)
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h_min = 0
    h_max = 100
    s_min = 0
    s_max = 255
    v_min = 20
    v_max = 255
    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imgHSV,lower,upper)
    imgResult = cv2.bitwise_and(img,img,mask=mask)

    img = 255 - cv2.medianBlur(img, 3)

    dst = cv2.Canny(img, 500, 50, None, 3)
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 50, 0)
    strong_lines = np.zeros([50, 1, 2])
    print(lines)
    if int(0 if lines is None else 1) == 0:
        cv2.putText(cdst, "No lines detected", (50, 50), 0, 1, (0, 0, 255), 1, 2)

    else:
        linedetection(lines,cdst)
    #cv2.imshow("Mask Images", mask)
    cv2.imshow("Original Images", img)
    cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)

    cv2.waitKey()
    cv2.waitKey(1)
