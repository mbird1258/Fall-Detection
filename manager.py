import utils
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

class CameraManager:
    def __init__(self, 
                 cap, 
                 OutputDirectoryName=None,
                 SaveVideoOfIncident=True, 
                 IncidentVideoLength=[2000,1000], 
                 TimeBeforeCullBody=250,
                 VelocityThreshold=0.05,
                 IncidentAccelThreshold=0.7,
                 CooldownBetweenIncidents=1000,
                 CamViewDepth = None):
        '''
        args:
            cap: cv2 videocapture of camera/video
            OutputDirectoryName: output directory name
            SaveVideoOfIncident: To save video of every incident or not
            IncidentVideoLength: If saving videos, how long before and after incident to save (milliseconds)
            TimeBeforeCullBody: How long of body not being detected before body is assumed to be gone (milliseconds)
            VelocityThreshold: How big a median distance between bodies' corresponding joints in 2 frames before determining body in frame 2 is not same (m/s)
            IncidentAccelThreshold: How high an acceleration/force on body before incident is reported/logged (m/s^2)
            CooldownBetweenIncidents: How long before another incident is reported (avoids 50 incidents reported in one fall) (milliseconds)
            CamViewDepth: Distance between camera and the plane in front of the camera that on which the image lies (distance to image plane) (pixels)
        '''

        self.cap = cap
        self.OutputDirectoryName = OutputDirectoryName if SaveVideoOfIncident==True else None
        path = f"Out/{self.OutputDirectoryName}/"
        if not os.path.exists(path): os.makedirs(path)

        self.SaveVideoOfIncident = SaveVideoOfIncident
        self.IncidentVideoLength = IncidentVideoLength if SaveVideoOfIncident==True else None
        self.TimeBeforeCullBody = TimeBeforeCullBody
        self.VelocityThreshold = VelocityThreshold
        self.IncidentAccelThreshold = IncidentAccelThreshold
        self.CooldownBetweenIncidents = CooldownBetweenIncidents

        self.AsyncUpdateVideoTimes = [] if SaveVideoOfIncident==True else None
        self.VideoNum = [0, 0] if SaveVideoOfIncident==True else None
        self.IncidentVideo = [] if SaveVideoOfIncident==True else None
        self.bodies = []
        self.log = log()
        self.PoseManager = utils.PoseManager(CamViewDepth)

    def UpdateBodies(self, img, FrameMS):
        '''
        method:
            get median distance between each bodies' joints
            if distance < thresh:
                bodies are matched
            else:
                body2 ≠ body1, create body2 as new body object
        
        returns:
            [body1, body2, ...]
            each entry(body) is an object containing the xyz, velocity, and acceleration of each joint
        '''
        
        if len(self.bodies) == 0:
            _, BodyCoordinates2 = self.PoseManager.GetBodyPose(img, FrameMS)
            for BodyCoords in BodyCoordinates2:
                self.bodies.append(body(BodyCoords, FrameMS))
            return
        
        BodyCoordinates1 = np.array([body.xyz for body in self.bodies]) # shape: [# bodies 1, # joints, 3]
        _, BodyCoordinates2 = self.PoseManager.GetBodyPose(img, FrameMS) # shape: [# bodies 2, # joints, 3]
        
        if len(BodyCoordinates2) == 0:
            return
        
        TimeLastUpdated1 = np.array([body.TimeLastUpdated for body in self.bodies]) # shape: [# bodies 1]
        dtArr = FrameMS - TimeLastUpdated1 # shape: [# bodies 1]

        VelocityArr = (BodyCoordinates1[:, np.newaxis, :, :] - BodyCoordinates2[np.newaxis, :, :, :])/dtArr[:, np.newaxis, np.newaxis, np.newaxis] # shape: [# bodies1, # bodies2, # joints, 3]
        MatchArr1 = np.nanmax(np.sqrt(np.sum(VelocityArr**2, axis=3)), axis=2)
        MatchArr2 = np.copy(MatchArr1)

        while True:
            ind1, ind2 = np.unravel_index(np.nanargmin(MatchArr1, axis=None), MatchArr1.shape)
            vel = VelocityArr[ind1, ind2]

            if MatchArr1[ind1, ind2] > self.VelocityThreshold:
                break

            self.bodies[ind1].UpdateData(BodyCoordinates2[ind2], vel, FrameMS)
            MatchArr1[ind1] = np.inf
            MatchArr1[:, ind2] = np.inf
            MatchArr2[:, ind2] = np.inf
        
        while True:
            ind1, ind2 = np.unravel_index(np.nanargmin(MatchArr2, axis=None), MatchArr2.shape)
            vel = VelocityArr[ind1, ind2]

            if MatchArr2[ind1, ind2] == np.inf:
                break

            self.bodies.append(body(BodyCoordinates2[ind2], FrameMS))
            MatchArr2[:, ind2] = np.inf

    def UpdateBodiesList(self, FrameMS):
        bodies = []

        for body in self.bodies:
            if FrameMS - body.TimeLastUpdated <= self.TimeBeforeCullBody:
                bodies.append(body)
        
        self.bodies = bodies

    def DetectIncident(self, FrameMS, verbose=False):
        FallsOccured = 0

        for body in self.bodies:
            if FrameMS - body.LastIncidentTime < self.CooldownBetweenIncidents or np.max(body.accel) <= self.IncidentAccelThreshold:
                continue
            
            body.UpdateIncident(FrameMS)
            FallsOccured += 1
            
            if self.SaveVideoOfIncident:
                self.AsyncUpdateVideoTimes.append(FrameMS + self.IncidentVideoLength[1])
                if (FallsOccured == 1): 
                    self.VideoNum[0] += 1
            
            # ==!== Other things to do if an incident is detected ==!== #
            if verbose:
                print(f"Time(s): {np.round(FrameMS/1000, 1)} - fall >.<")
        
        if self.SaveVideoOfIncident:
            return FallsOccured
        
        return FallsOccured

    def StoreVid(self, path, video):
        height, width, _ = video[0].shape

        out = cv2.VideoWriter(path, 
                              cv2.VideoWriter_fourcc(*'mp4v'), 
                              self.cap.get(cv2.CAP_PROP_FPS), 
                              (width,height))

        for img in video:
            out.write(img)

        out.release()

    def UpdateVideoAsync(self, FrameMS, force=False):
        if force:
            for _ in self.AsyncUpdateVideoTimes:
                path = f"Out/{self.OutputDirectoryName}/{self.VideoNum[1]}.mp4"
                self.StoreVid(path, self.IncidentVideo)

                self.VideoNum[1] += 1
            
            self.AsyncUpdateVideoTimes = []

            return
        
        mask = np.ones_like(self.AsyncUpdateVideoTimes).astype(np.bool_)

        for ind, time in enumerate(self.AsyncUpdateVideoTimes):
            if FrameMS > time:
                path = f"Out/{self.OutputDirectoryName}/{self.VideoNum[1]}.mp4"
                self.StoreVid(path, self.IncidentVideo)

                self.VideoNum[1] += 1
                mask[ind] = False
                continue
        
        self.AsyncUpdateVideoTimes = np.array(self.AsyncUpdateVideoTimes)[mask].tolist()

    def main(self):
        '''
        returns log object of incident data
        '''
        
        res, img = self.cap.read()
        if not res:
            if self.SaveVideoOfIncident:
                # If video terminates, save video of incident as is even if not full n milliseconds after passed
                self.UpdateVideoAsync(None, True)
            return False, self.log
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # If saving video of incidents, keep a running video of the past n milliseconds to save when an incident occurs
        if self.SaveVideoOfIncident:
            self.IncidentVideo.append(img)
            self.IncidentVideo = self.IncidentVideo[-int(self.IncidentVideoLength[0]/1000*self.cap.get(cv2.CAP_PROP_FPS)):]

        FrameMS = self.cap.get(cv2.CAP_PROP_POS_MSEC) # self.cap.get(cv2.CAP_PROP_POS_FRAMES)*self.cap.get(cv2.CAP_PROP_FPS)
        if self.cap.get(cv2.CAP_PROP_POS_FRAMES) == 1:
            FrameMS = 0
        
        self.UpdateBodies(img, FrameMS)
        self.UpdateBodiesList(FrameMS)

        NumFalls = self.DetectIncident(FrameMS)
        if self.SaveVideoOfIncident:
            path = f"Out/{self.OutputDirectoryName}/{self.VideoNum[0]}.mp4"
            self.log(FrameMS, NumFalls, path)
        else:
            self.log(FrameMS, NumFalls)

        if self.SaveVideoOfIncident:
            self.UpdateVideoAsync(FrameMS)
        
        return True, self.log


class body:
    def __init__(self, xyz, FrameMS):
        self.xyz = xyz # shape: [# joints, 3 (dx/dt, dy/dt, dz/dt)]
        self.vel = None # shape: [# joints, 3 (dx/dt, dy/dt, dz/dt)]
        self.accel = np.zeros(xyz.shape[0]) # shape: [# joints]
        self.TimeLastUpdated = FrameMS
        self.LastIncidentTime = -np.inf

    def UpdateData(self, xyz2, vel2, FrameMS):
        vel1 = self.vel
        vel2 = np.copy(vel2)
        
        t1 = self.TimeLastUpdated
        t2 = FrameMS
        dt = (t2-t1)/1000

        if type(vel1) == np.ndarray:
            self.accel = np.sqrt(np.sum((vel2-vel1)**2, axis=1))/dt
            self.accel = np.nan_to_num(self.accel)
        
        self.xyz = xyz2
        self.vel = vel2
        self.TimeLastUpdated = t2

        return self

    def UpdateIncident(self, FrameMS):
        self.LastIncidentTime = FrameMS


class log:
    def __init__(self):
        self.contents = []
    
    def __call__(self, FrameMS, NumFalls, VideoPath=False):
        if NumFalls == 0:
            return

        if VideoPath:
            self.contents.append((FrameMS, NumFalls, VideoPath))
            return
        
        self.contents.append((FrameMS, NumFalls))
    
    def AccessVideoPathByInd(self, ind):
        return self.contents[ind][2]
    
    def __str__(self):
        if len(self.contents) == 0:
            return "No Contents"
        
        out = ""
        for ind, entry in enumerate(self.contents):
            out = out+f"\nIndex: {ind} || Time(s): {np.round(entry[0]/1000, 2)} || Num of falls: {entry[1]} || Recorded Videos: {len(entry) == 3}"
        
        return out
