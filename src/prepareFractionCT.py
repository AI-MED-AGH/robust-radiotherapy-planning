import numpy as np
import cv2
import glob
import pydicom
import gdcm
import os
import sys
import nibabel as nib
from skimage.transform import rescale,resize
from scipy.ndimage import rotate
from scipy.interpolate import interpn
from scipy.ndimage import distance_transform_edt
import SimpleITK as sitk
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.ndimage import label
from scipy.ndimage import distance_transform_edt

StructureDICOM_SOPClassUID = "1.2.840.10008.5.1.4.1.1.481.3"
DoseDICOM_SOPClassUID ='1.2.840.10008.5.1.4.1.1.481.2'

def readDICOM3D(DICOM_DIR):

    dicomFiles = [file for file in os.listdir(DICOM_DIR) if file.endswith('dcm') and file.startswith('CT')]
    dicoms = []
    for file in dicomFiles:
        ds = pydicom.dcmread(DICOM_DIR+file)
        dicoms.append((DICOM_DIR+file,int(ds[0x0020,0x0013].value)))
    dicoms = sorted(dicoms, key=lambda x: x[1])

    im3D = []
    slicePositions = []
    for d in dicoms:
        ds = pydicom.dcmread(d[0])
        rescaleIntercept = ds[0x0028,0x1052].value
        rescaleSlope = ds[0x0028,0x1053].value
        slicePositions.append(ds.ImagePositionPatient[2])
        pix = np.copy(ds.pixel_array)
        pix = pix*rescaleSlope
        pix += int(rescaleIntercept)
        im3D.append(pix)
    
    im3D = np.asarray(im3D,dtype=np.int16)
    im3D = np.swapaxes(im3D,0,1)
    im3D = np.swapaxes(im3D,1,2)

    ds = pydicom.dcmread(dicoms[0][0])
#https://dicom.innolitics.com/ciods/ct-image/image-plane/00200032    
#ImagePositionPatient specifies coordinates of the upper left hand corner of the image: it is the center of the first voxel transmitted
    CTOrigin = ds.ImagePositionPatient
    CTPixelSize = ds.PixelSpacing
    CTSliceThickness = ds.SliceThickness

    grid = (np.arange(CTOrigin[1],CTOrigin[1]+CTPixelSize[1]*im3D.shape[0],CTPixelSize[1]),
            np.arange(CTOrigin[0],CTOrigin[0]+CTPixelSize[0]*im3D.shape[1],CTPixelSize[0]),
            np.asarray(slicePositions,dtype=np.float64))
    
    return grid,im3D

def displayStructureIDs(structuresFile):
    ROINumbers = []
    ds = pydicom.dcmread(structuresFile)
    for nstruct,struct in enumerate(ds.StructureSetROISequence):
        #print(struct.ROINumber,struct.ROIName)
        ROINumbers.append((struct.ROINumber,struct.ROIName))
    return ROINumbers

# Creates volume corresponding to a structure with structID
# Returns 3D image with black pixels correcsponding to background and pixels labeled with ds.StructureSetROISequence[structID].ROINumber 
# corresponding to the structure of interest
def drawContours(imSize,roiID,pathToCT,structuresDicomFileName):

    ds = pydicom.dcmread(structuresDicomFileName)

    roiNumbers = [dum.ROINumber for dum in ds.StructureSetROISequence]
    structID = roiNumbers.index(roiID)
    
    #print(ds.StructureSetROISequence[structID].ROINumber,ds.StructureSetROISequence[structID].ROIName)

    assert ds.SOPClassUID == StructureDICOM_SOPClassUID, "This is not a Dicom Structure file"

    #ROINumber and ROIName are defined in ds.StructureSetROISequence
    #ROINumber is referenced to in ds.ROIContourSequence
    #Each ROI in ds.StructureSetROISequence correpsponds to a contour sequence in ds.ROIContourSequence
    #They must be matched based on ROINumber, referenced to in ds.ROIContourSequence and defined in ds.StructureSetROISequence
    #There is a single contour sequence corresponding to a ROI specified by structID - I extract this contour sequence

    ROI = [ds.ROIContourSequence[u].ContourSequence for u in range(len(ds.ROIContourSequence)) if ds.ROIContourSequence[u].ReferencedROINumber 
           == ds.StructureSetROISequence[structID].ROINumber][0]

    positions = []
    CTs = glob.glob(pathToCT + '/CT*.dcm')
    for fname in CTs:
        ctds = pydicom.dcmread(fname)
        positions.append((ctds.ImagePositionPatient[2],fname)) 
    positions = sorted(positions, key=lambda x: x[0])  
    delta = positions[1][0] - positions[0][0]

    dum = np.zeros(imSize,dtype=np.uint8)
    
    for seq in ROI:
        points = np.swapaxes(np.reshape(seq.ContourData,(-1,3)),0,1)
        pos = points[2,0]
        for p in positions:
            if abs(pos - p[0]) < delta/2:
                fname = p[1]
                
        dicImage = pydicom.dcmread(fname)
        
        M = np.zeros((3,3),dtype = np.float32)
        M[0,0] = dicImage[0x0020, 0x0037].value[1]* dicImage[0x0028, 0x0030].value[0]
        M[1,0] = dicImage[0x0020, 0x0037].value[0]* dicImage[0x0028, 0x0030].value[0]
        M[0,1] = dicImage[0x0020, 0x0037].value[4]* dicImage[0x0028, 0x0030].value[1]
        M[1,1] = dicImage[0x0020, 0x0037].value[3]* dicImage[0x0028, 0x0030].value[1]
        M[0,2] = dicImage[0x0020, 0x0032].value[0]
        M[1,2] = dicImage[0x0020, 0x0032].value[1]
        M[2,2] = 1.0
        M = np.linalg.inv(M)
        points[2,:].fill(1)
        points = np.dot(M,points)[:2,:]

        big = int(ds.StructureSetROISequence[structID].ROINumber) # 255
        CTSlice = int(dicImage[0x0020,0x0013].value)-1            # numery sliców w Dicom startują od 1
        dum2D = np.zeros(imSize[0:2],dtype=np.uint8)
        for id in range(points.shape[1]-1):
            cv2.line(dum2D,(int(points[1,id]),int(points[0,id])),(int(points[1,id+1]),int(points[0,id+1])),big,1)
        cv2.line(dum2D,(int(points[1,points.shape[1]-1]),int(points[0,points.shape[1]-1])),(int(points[1,0]),int(points[0,0])),big,1)

        dum[dum2D!=0,CTSlice] = dum2D[dum2D!=0]
        
    for sl in range(dum.shape[2]):
        im_flood_fill = dum[...,sl].copy()
        h, w = dum.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        im_flood_fill = im_flood_fill.astype("uint8")
        cv2.floodFill(im_flood_fill, mask, (0, 0), 128)
        dum[im_flood_fill!=128,sl] = big
        
    return dum

def readTPSDose(filename):

    ds = pydicom.dcmread(filename)

    assert ds.SOPClassUID==DoseDICOM_SOPClassUID, 'This is not a Dicom DOSE file!'

    dose = ds.pixel_array
    scaling = float(ds.DoseGridScaling)

    dose = np.swapaxes(dose,0,1)
    dose = np.swapaxes(dose,1,2)
    dose = np.asarray(dose,dtype=np.float64)*scaling

    print(dose.shape)

    doseOrigin = ds.ImagePositionPatient
    dosePixelSize = ds.PixelSpacing
    doseSlicePositions = ds[0x3004, 0x000c].value

    grid = (np.arange(doseOrigin[1],doseOrigin[1]+dosePixelSize[1]*dose.shape[0],dosePixelSize[1]),
        np.arange(doseOrigin[0],doseOrigin[0]+dosePixelSize[0]*dose.shape[1],dosePixelSize[0]),
        np.asarray([doseOrigin[2]+x for x in doseSlicePositions],dtype=np.float64))

    return grid,dose


if __name__ == "__main__":


    SAVE_DIR = './DATA/'

    startDir = '/data/Shared/sco_PreciseART_Pacjenci_kV/'

    patients = sorted(glob.glob(startDir + 'Pacjent_' + sys.argv[1] + '_*'))

    f = open('prostaty.txt','r')
    lines = f.readlines()
    f.close()
    mapping = {l.split()[0]:l.split()[1] for l in lines}


    fractions_to_exclude = []
    for patient in patients:
        patient_id = os.path.basename(patient).split('_')[1]
        workDirs = sorted(glob.glob(patient + '/ANON-PATIENT_ID_001/Frakcja*'))
        for n,workDir in enumerate(workDirs):

            fraction_id = os.path.basename(workDir).split('_')[-1]

            if fraction_id in fractions_to_exclude:
                continue

            print(workDir)
       
            CT_DicomDir = workDir + '/'

            gridCT,ct = readDICOM3D(CT_DicomDir)

            if n==0:
                ctShape = ct.shape[:2]
                aff = np.eye(4)
                aff[0,0] = abs(gridCT[0][1]-gridCT[0][0])
                aff[1,1] = abs(gridCT[1][1]-gridCT[1][0])
                aff[2,2] = abs(gridCT[2][1]-gridCT[2][0])

            if n>0:
                if ct.shape[:2] != ctShape:
                    ct = resize(ct,ctShape + (ct.shape[2],),anti_aliasing = True, preserve_range = True)

            m = np.min(ct)
            if m < -1024:
                ct[ct==m] = -1024
            m = np.max(ct)
            if m > 1500:
                ct[ct>1500] = 1500

            os.makedirs(f'{SAVE_DIR}/CT/Patient_{patient_id}',exist_ok=True)
            baseName = 'Patient_' + patient_id + '_fraction_' + fraction_id + '_.nii.gz'
            niftiImage = nib.Nifti1Image(ct, affine=aff)
            nib.save(niftiImage,f'{SAVE_DIR}/CT/Patient_{patient_id}/{baseName}')  

            structFiles = glob.glob(workDir + '/RTS*.dcm')
            if len(structFiles)==1:
                TPS_StructDicomFileName = structFiles[0]
                rois = displayStructureIDs(TPS_StructDicomFileName)

                print(rois)

                flagSkin = 0
                for roi in rois:
                    if roi[1].lower() in ['skin','skin_','external_roi','external roi','external roi_2','external roi_1','external']:
                        nstruct = roi[0]
                        skin = drawContours(ct.shape,nstruct,CT_DicomDir,TPS_StructDicomFileName)
                        skin[skin!=0] = 1
                        flagSkin = 1
                        break

                flagProstate = 0
                prostateName = mapping[patient_id]
                for roi in rois:
                    if roi[1].lower() == prostateName.lower():
                        nstruct = roi[0]
                        prostate = drawContours(ct.shape,nstruct,CT_DicomDir,TPS_StructDicomFileName)
                        prostate[prostate!=0] = 1
                        flagProstate = 1
                        break

                flagBladder = 0
                flagRectum = 0
                flagFemurL = 0
                flagFemurR = 0
                for roi in rois:
                    if roi[1].lower() in ['bladder','pecherz','bladder_contrast']:
                        nstruct = roi[0]
                        bladder = drawContours(ct.shape,nstruct,CT_DicomDir,TPS_StructDicomFileName)
                        flagBladder = 1
                        break
                for roi in rois:
                    if roi[1].lower() in ['rectum','rectum1','rectum2','odbytnica']:
                        nstruct = roi[0]
                        rectum = drawContours(ct.shape,nstruct,CT_DicomDir,TPS_StructDicomFileName)
                        flagRectum = 1
                        break
                for roi in rois:
                    if roi[1] in ['FEMUR_HEAD_L','glowa l','GLOWKA LEWA','PROT_F_HEAD_L','proteza','FEMUR_L_PROTEZA']:
                        nstruct = roi[0]
                        femurL = drawContours(ct.shape,nstruct,CT_DicomDir,TPS_StructDicomFileName)
                        flagFemurL = 1
                        break
                for roi in rois:
                    if roi[1] in ['FEMUR_HEAD_R','glowa p','GLOWKA PRAWA','PROT_F_HEAD_R','proteza','FEMUR_R_PROTEZA']:
                        nstruct = roi[0]
                        femurR = drawContours(ct.shape,nstruct,CT_DicomDir,TPS_StructDicomFileName)
                        flagFemurR = 1
                        break

                print('flags',flagSkin,flagBladder,flagRectum,flagProstate,flagFemurL,flagFemurR)
                

                struct = np.zeros(prostate.shape,dtype=np.uint8)
                struct[skin!=0] = 1
                struct[rectum!=0] = 2
                struct[bladder!=0] = 3
                struct[prostate!=0] = 4
                struct[femurL!=0] = 5
                struct[femurR!=0] = 5

                if n>0:
                    if ct.shape[:2] != ctShape:
                        struct = resize(struct,ctShape + (struct.shape[2],),anti_aliasing = True, preserve_range = True, order = 0)
                        struct = np.asarray(struct,dtype=np.uint8)

                os.makedirs(f'{SAVE_DIR}/STRUCTURES/Patient_{patient_id}',exist_ok=True)
                baseName = 'Patient_' + patient_id + '_fraction_' + fraction_id + '_.nii.gz'
                niftiImage = nib.Nifti1Image(struct, affine=aff)
                nib.save(niftiImage,f'{SAVE_DIR}/STRUCTURES/Patient_{patient_id}/{baseName}')  

            if n==0:
                doseFiles = glob.glob(workDir + '/RTDose*.dcm')
                if len(doseFiles)==1:
                    DoseDicomFileName = doseFiles[0]
                    _, dose = readTPSDose(DoseDicomFileName)

                    os.makedirs(f'{SAVE_DIR}/DOSES/Patient_{patient_id}',exist_ok=True)
                    baseName = 'Patient_' + patient_id + '_fraction_' + fraction_id + '_.nii.gz'
                    niftiImage = nib.Nifti1Image(dose, affine=aff)
                    nib.save(niftiImage,f'{SAVE_DIR}/DOSES/Patient_{patient_id}/{baseName}')  

            #break

        #break
