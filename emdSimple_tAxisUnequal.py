#Based on https://www.mathworks.com/help/signal/ref/emd.html

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d, Akima1DInterpolator
import pandas as pd


class Emd():
    def __init__(self, seq, tAxis = [], testMode = False):
        self.testMode = testMode #visualize, print everything
        
        self.seq = seq
        self.length = len(seq)
        self.tAxis = tAxis
        if self.tAxis == []:
            self.tAxis = np.linspace(0, self.length-1, self.length)
            
        self.siftMethod = self.cubic_sift
        #self.siftMethod = self.extremumCenter_sift
        #self.siftMethod = self.akimaSpline_sift
        
        self.SiftRelativeTolerance = 0.06
        self.SiftMaxIterations = 80
        
        self.maxEnergyRatio = 30
        self.maxNumExtrema = 2
        self.maxNumIMF = 10
        
        self.IMFs = []
        self.finalRes = None


    def cubic_sift(self, r_i_Prev):
        upper_peaks, _ = find_peaks(r_i_Prev)
        lower_peaks, _ = find_peaks(-r_i_Prev)
        if len(lower_peaks) < 4 or len(upper_peaks) < 4:
            return r_i_Prev
        
        f1 = interp1d([self.tAxis[x] for x in upper_peaks],r_i_Prev[upper_peaks], kind = 'cubic', bounds_error=False, fill_value=(r_i_Prev[upper_peaks][0], r_i_Prev[upper_peaks][-1]))
        f2 = interp1d([self.tAxis[x] for x in lower_peaks],r_i_Prev[lower_peaks], kind = 'cubic', bounds_error=False, fill_value=(r_i_Prev[lower_peaks][0], r_i_Prev[lower_peaks][-1]))

        y1 = f1(self.tAxis)
        y2 = f2(self.tAxis)
  
        avg_envelope = (y1 + y2) / 2

        r_i_Cur = r_i_Prev - avg_envelope
        if self.testMode:
            plt.plot(self.tAxis, r_i_Prev)
            plt.plot(self.tAxis, y1)
            plt.plot(self.tAxis, y2)
            plt.plot(self.tAxis, avg_envelope)
            plt.title('cubic')
            plt.show()
        return r_i_Cur
    
    def extremumCenter_sift(self, r_i_Prev):
        upper_peaks, _ = find_peaks(r_i_Prev)
        lower_peaks, _ = find_peaks(-r_i_Prev)
        f1 = interp1d([self.tAxis[x] for x in upper_peaks],r_i_Prev[upper_peaks], kind = 'linear', bounds_error=False, fill_value=(r_i_Prev[upper_peaks][0], r_i_Prev[upper_peaks][-1]))
        f2 = interp1d([self.tAxis[x] for x in lower_peaks],r_i_Prev[lower_peaks], kind = 'linear', bounds_error=False, fill_value=(r_i_Prev[lower_peaks][0], r_i_Prev[lower_peaks][-1]))
    
        
        extremumPts = np.concatenate((upper_peaks,lower_peaks))
        extremumPts.sort()
        
        if len(extremumPts) < 4:
            return r_i_Prev
        
        extremumPtsTAXIS = [self.tAxis[x] for x in extremumPts]
        y1 = f1(extremumPtsTAXIS)
        y2 = f2(extremumPtsTAXIS)
        
        extremumCenter = (y1 + y2) / 2

        f = interp1d(extremumPtsTAXIS,extremumCenter, kind = 'cubic', bounds_error=False, fill_value=(extremumCenter[0], extremumCenter[-1]))
        
        avg_envelope = f(self.tAxis)
        r_i_Cur = r_i_Prev - avg_envelope
        
        if self.testMode:
            plt.plot(self.tAxis, r_i_Prev)
            plt.plot(self.tAxis, f1(self.tAxis))
            plt.plot(self.tAxis, f2(self.tAxis))
            plt.plot(self.tAxis, avg_envelope)
            plt.title('extremumCenter')
            plt.show()
            
        return r_i_Cur
    
    def akimaSpline_sift(self, r_i_Prev):
        upper_peaks, _ = find_peaks(r_i_Prev)
        lower_peaks, _ = find_peaks(-r_i_Prev)
        
        if len(lower_peaks) < 4 or len(upper_peaks) < 4:
            return r_i_Prev
        
        f1 = Akima1DInterpolator([self.tAxis[x] for x in upper_peaks],r_i_Prev[upper_peaks])
        f2 = Akima1DInterpolator([self.tAxis[x] for x in lower_peaks],r_i_Prev[lower_peaks])
    
        y1 = f1(self.tAxis)
        y2 = f2(self.tAxis)
  
        avg_envelope = (y1 + y2) / 2
        
        i = 0
        if np.isnan(avg_envelope[i]):
            while np.isnan(avg_envelope[i]):
                i += 1
            for j in range(i):
                avg_envelope[j] = avg_envelope[i]
        
        i = self.length-1
        if np.isnan(avg_envelope[i]):
            while np.isnan(avg_envelope[i]):
                i -= 1
            for j in range(i+1, self.length):
                avg_envelope[j] = avg_envelope[i] 

        if self.testMode:
            plt.plot(self.tAxis, r_i_Prev)
            plt.plot(self.tAxis, y1)
            plt.plot(self.tAxis, y2)
            plt.plot(self.tAxis, avg_envelope)
            plt.title('akima')
            plt.show()
        
        r_i_Cur = r_i_Prev - avg_envelope
        return r_i_Cur
    
        
        
        
    def findOneIMF(self, r_i):
        r_i_Prev = r_i
        IN = 0
        
        while True:
            r_i_Cur = self.siftMethod(r_i_Prev)
            IN += 1
            
            if self.testMode:
                plt.figure(figsize = (20,6))
                plt.subplot(1,2,1)
                plt.plot(self.tAxis,r_i_Prev)
                plt.xlabel('Time [s]')
                plt.title('r_i_Prev' + str(IN-1))
                plt.subplot(1,2,2)
                plt.plot(self.tAxis,r_i_Cur)
                plt.xlabel('Time [s]')
                plt.title('r_i_Cur' + str(IN))
                plt.show()
            
            if IN > self.SiftMaxIterations:
                self.IMFs.append(r_i_Cur)
                break
            if IN > 1:
                RT = np.linalg.norm(r_i_Prev-r_i_Cur)**2/ np.linalg.norm(r_i_Prev)**2
                if self.testMode:
                    print('IN = ' + str(IN)+ ' /// RT = ' +str(RT))
                if RT < self.SiftRelativeTolerance:
                    self.IMFs.append(r_i_Cur)
                    break
            
            r_i_Prev = r_i_Cur
            
    def findIMFs(self):
        r_i = self.seq
        while True:
            upper_peaks, _ = find_peaks(r_i)
            lower_peaks, _ = find_peaks(-r_i)
            if len(upper_peaks) + len(lower_peaks) < self.maxNumExtrema:
                self.finalRes = r_i
                break
            
            ER = 10*np.log10(np.linalg.norm(self.seq)**2/ np.linalg.norm(r_i)**2)
            if self.testMode:
                print('numOfIMF = ' + str(len(self.IMFs)) +' /// ER = '+str(ER))
            if ER > self.maxEnergyRatio:
                self.finalRes = r_i
                break
            
            if len(self.IMFs) >= self.maxNumIMF:
                self.finalRes = r_i
                break
            
            self.findOneIMF(r_i)
            r_i = r_i - self.IMFs[-1]
         
    def printAll(self):
        for i in range(len(self.IMFs)):
            plt.plot(self.tAxis, self.IMFs[i])
            plt.xlabel('Time [s]')
            plt.title('Imf ' + str(i+1))
            plt.show()
        plt.plot(self.tAxis, self.finalRes)
        plt.xlabel('Time [s]')
        plt.title('Res, Method: ' + self.siftMethod.__name__)
        plt.show()
        
        plt.plot(self.tAxis, self.seq-self.finalRes)
        plt.xlabel('Time [s]')
        plt.title('Seq - Res')
        plt.show()
  

def uniformRandom():
    # Generate random signal
    np.random.seed(0) # For reproducibility
    sequence = np.random.randn(1000) + np.linspace(0,50, 1000)
    plt.plot(sequence)
    plt.title('uniformRandom')
    plt.show() 
    emdObj = Emd(sequence, testMode=False)
    emdObj.findIMFs()   
    emdObj.printAll()   
        
        
def sinSample():
    tAxis = np.arange(0, 1000, 1)
    sequence = np.sin(tAxis) + np.sin(np.sqrt(13)*tAxis) +np.sin(np.sqrt(23)*tAxis) + 1/50*tAxis
    plt.plot(sequence)
    plt.title('sinSample')
    plt.show()
             
    emdObj = Emd(sequence, testMode=False)
    emdObj.findIMFs()   
    emdObj.printAll()   

def sinSampleUnequal():
    np.random.seed(0)
    tAxis =  np.asarray(sorted(list(set(np.random.randint(1000, size=500)))))
    print(len(tAxis), max(tAxis))
    sequence = np.sin(tAxis) + np.sin(np.sqrt(13)*tAxis) +np.sin(np.sqrt(23)*tAxis) 
    sequence = sequence + 1/50*tAxis
    plt.plot(tAxis, sequence)
    plt.title('sinSampleUnequal')
    plt.show()
             
    emdObj = Emd(sequence, tAxis=tAxis, testMode=False)
    emdObj.findIMFs()   
    emdObj.printAll()   

def bitcoin():
    
    # https://www.kaggle.com/datasets/kognitron/zielaks-bitcoin-historical-data-wo-nan
    
    nRowsRead = 10000 # specify 'None' if want to read whole file
    
    #total 3330541

    df = pd.read_csv('bitstamp_cleaned.csv', delimiter=',', nrows = nRowsRead)
    df.dataframeName = 'bitstamp_cleaned.csv'
    nRow, nCol = df.shape
    print(f'There are {nRow} rows and {nCol} columns')
    print(df.head(5))
    print(df.info())

    sequence = df.loc[:,'Weighted_Price'].values
    tAxis = df.loc[:, 'Unix_Timestamp'].values
    
    plt.plot(tAxis, sequence)
    plt.title('sinSampleUnequal')
    plt.show()
    
    emdObj = Emd(sequence, tAxis=tAxis, testMode=False)
    emdObj.findIMFs()   
    emdObj.printAll()   

bitcoin()