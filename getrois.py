import numpy as np


def GetROIs(I1, ROInum, pixsize, roisize, ROImethod):
    if ROImethod == 'Manual':
        roixc = [round(pixsize / 2)]
        roiyc = [round(pixsize / 2)]
    elif ROImethod == 'Intensity':
        k = 0
        m = 0
        n = 0
        roixc = np.zeros(ROInum)
        roiyc = np.zeros(ROInum)
        I2 = np.zeros((pixsize, pixsize))
        for i in range(pixsize):
            for j in range(pixsize):
                if i == 1 and j == 1:
                    I2[i, j] = n
                elif i == 1 and j != 1:
                    if I1[i, j] == I1[i, j - 1]:
                        I2[i, j] = I2[i, j - 1]
                    else:
                        n = n + 1
                        I2[i, j] = n
                elif i != 1 and j == 1:
                    if I1[i, j] == I1[i - 1,j]:
                        I2[i, j] = I2[i - 1, j]
                    elif I1[i, j] == I1[i - 1, j + 1]:
                        I2[i, j] = I2[i - 1, j + 1]
                    else:
                        n = n + 1
                        I2[i, j] = n
                elif i != 1 and j != 1 and j != pixsize - 1:
                    if I1[i, j] == I1[i - 1, j - 1]:
                        I2[i, j] = I2[i - 1, j - 1]
                    elif I1[i, j] == I1[i - 1, j]:
                        I2[i, j] = I2[i - 1, j]
                    elif I1[i, j] == I1[i - 1, j + 1]:
                        I2[i, j] = I2[i-1, j + 1]
                    elif I1[i, j] == I1[i, j - 1]:
                        I2[i, j] = I2[i, j - 1]
                    else:
                        n = n + 1
                        I2[i, j] = n
                elif i != 1 and j == pixsize - 1:
                    if I1[i, j] == I1[i - 1, j - 1]:
                        I2[i, j] = I2[i - 1, j - 1]
                    elif I1[i, j] == I1[i - 1, j]:
                        I2[i, j] = I2[i - 1, j]
                    elif I1[i, j] == I1[i, j - 1]:
                        I2[i, j] = I2[i, j - 1]
                    else:
                        n = n + 1
                        I2[i, j] = n
        I2 = I2 / (1000 * I2.max())
        I1 = I1 + I2
        I = np.flipud(np.unique(I1))
        
        while k <= ROInum:
            print(k, m, n)
            row, col = np.where(I1 == I[m])
            x = round((max(col) + min(col)) / 2)
            y = round((max(row) + min(row)) / 2)
            if (x < (np.ceil(roisize / 2) + 1)) or ((pixsize - x) < (np.ceil(roisize / 2) + 1)) or (y < (np.ceil(roisize / 2) + 1)) or ((pixsize - y) < (np.ceil(roisize / 2) + 1)):
                m = m + 1
            else:
                for l in range(ROInum):
                    if (abs(roixc[l] - x) < (pixsize / 10)) and (abs(roiyc[l] - y) < (pixsize / 10)):
                        m = m + 1
                        break
                    elif l == ROInum:
                        roixc[k] = x
                        roiyc[k] = y
                        m = m + 1
                        k = k + 1

    elif ROImethod == 'Grid':
        edgeCount = round(np.sqrt(ROInum))
        edgeSpacing = round(roisize / 2 + 0.1 * pixsize)
        edgePoints = np.linspace(edgeSpacing, pixsize - edgeSpacing, edgeCount)
        
        # roixc, roiyc = np.meshgrid(edgePoints)
        roixc, roiyc = np.meshgrid(edgePoints, edgePoints, indexing="xy")
        roixc = roixc.T.conj()
        roiyc = roiyc.T.conj()

    elif ROImethod == 'Radial':
        roixc = np.zeros(ROInum)
        roiyc = np.zeros(ROInum)
        roixc[0] = round(pixsize / 2)
        roiyc[0] = round(pixsize / 2)
        row1num = int(np.ceil((ROInum - 1) / 3))
        row2num = int(ROInum - 1 - row1num)
        rad2 = np.floor((pixsize - 2 - roisize) / 2.5)
        rad1 = round(rad2 / 2)
        ang1 = 2 * np.pi / row1num
        ang2 = 2 * np.pi / row2num
        
        for i in range(row1num):
            dx = rad1 * np.cos(i * ang1)
            dy = rad1 * np.sin(i * ang1)
            roixc[i + 1] = roixc[0] + dx
            roiyc[i + 1] = roiyc[0] + dy

        for i in range(row2num):
            dx = rad2 * np.cos(i * ang2)
            dy = rad2 * np.sin(i * ang2)
            roixc[i + 1 + row1num] = roixc[0] + dx
            roiyc[i + 1 + row1num] = roiyc[0] + dy

    elif ROImethod == 'Annular':
        roixc = np.zeros(ROInum)
        roiyc = np.zeros(ROInum)
        roixc[0] = round(pixsize / 2)
        roiyc[0] = round(pixsize / 2)
        angSpacing = 2 * np.pi / (ROInum - 1)
        radius = np.floor((pixsize - roisize) / 3)
        for i in range(ROInum - 1):
            dx = radius * np.cos(i * angSpacing)
            dy = radius * np.sin(i * angSpacing)
            roixc[i + 1] = roixc[0] + dx
            roiyc[i + 1] = roiyc[0] + dy
    return (roixc, roiyc)

if __name__ == "__main__":
    import ebsd_pattern
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    p = ebsd_pattern.get_pattern_file_obj("D:/Research/R2S10S5/Data/361.up2").read_data()
    nrows = p.nFileRows
    ncols = p.nFileCols
    npats = p.nPatterns
    print(npats)
    p = p.patterns
    
    I1 = p[npats // 2]
    ROInum = 15
    ROImethod = "Grid"
    pixsize = I1.shape[0]
    roisize = round(pixsize * 0.15)
    out = GetROIs(I1, ROInum, pixsize, roisize, ROImethod)
    
    # Plot image and draw on rectangles for the ROIs
    fig, ax = plt.subplots()
    ax.imshow(I1, cmap="gray")
    out = [out[0].flatten(), out[1].flatten()]
    for i in range(len(out[0])):
        r = Rectangle((out[0][i] - roisize / 2, out[1][i] - roisize / 2), roisize, roisize, edgecolor="red", facecolor="none")
        ax.add_patch(r)
    plt.show()