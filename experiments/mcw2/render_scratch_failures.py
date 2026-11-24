"""Render last 7.2s of each paired replay from recorded SUMO positions."""
import json
import math
import subprocess
from pathlib import Path
from xml.etree import ElementTree
import cv2
import imageio_ffmpeg
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"results/scratch_failure_video_20261124"
LANES=[]
for lane in ElementTree.parse(ROOT/"scenarios/hv_high15000_entryprep/network.net.xml").iter("lane"):
    pts=np.array([[float(n) for n in p.split(",")[:2]] for p in lane.get("shape","").split()])
    if len(pts)>1: LANES.append((pts,pts.min(axis=0),pts.max(axis=0)))

def panel(scene,frame,foe):
    canvas=np.full((416,460,3),26,np.uint8)
    av=frame["av"]
    center=np.array(av["xy"])
    angle=math.radians(av["angle"])
    transform=4.4*np.array([[math.cos(angle),-math.sin(angle)],[-math.sin(angle),-math.cos(angle)]])
    def pixels(points):return ((points-center)@transform.T+[230,272]).astype(np.int32)
    for pts,lo,hi in LANES:
        if np.all(lo<center+85) and np.all(hi>center-85):
            line=pixels(pts)
            cv2.polylines(canvas,[line],False,(87,83,82),14,cv2.LINE_AA)
            cv2.polylines(canvas,[line],False,(112,105,100),1,cv2.LINE_AA)
    for car in frame["near"]+[av]:
        a=math.radians(car["angle"])
        tip=np.array(car["xy"])
        back=car["length"]*np.array([math.sin(a),math.cos(a)])
        side=.5*car["width"]*np.array([math.cos(a),-math.sin(a)])
        pts=pixels(np.array([tip+side,tip-side,tip-back-side,tip-back+side]))
        color=(240,211,51) if car is av else (72,75,252) if car["id"]==foe else (115,185,188)
        cv2.fillConvexPoly(canvas,pts,color,cv2.LINE_AA)
    cv2.rectangle(canvas,(0,0),(460,89),(32,32,32),-1)
    for n,t in enumerate([scene["model"]+" | "+scene["route"]+" | "+scene["outcome"], f'{av["edge"]}  lane {av["lane"]}  {av["speed"]*3.6:.1f} km/h',
                           f'to end {(scene["result"]["steps"]-frame["step"])*.2:.1f}s   accel {frame["accel"]:+.2f}   lane cmd {frame["cmd"]}']):
        cv2.putText(canvas,t,(9,23+27*n),cv2.FONT_HERSHEY_SIMPLEX,.55,(235,235,235),1,cv2.LINE_AA)
    return canvas

def main():
    files=sorted(OUT.glob("dagger2-route*.json"))
    ff=imageio_ffmpeg.get_ffmpeg_exe()
    movie=OUT/"paired-replays.mp4"
    sheets=[]
    with subprocess.Popen([ff,"-loglevel","error","-y","-f","rawvideo","-pix_fmt","bgr24","-s","920x416","-r","5","-i","-","-vcodec","libx264","-pix_fmt","yuv420p","-movflags","+faststart",str(movie)],stdin=subprocess.PIPE) as proc:
        for f in files:
            pair=[json.loads(p.read_bytes()) for p in [f,OUT/f.name.replace("dagger2","ppo_16")]]
            sequence=[]
            for i in range(-36,0):
                both=np.hstack([panel(s,s["frames"][max(i,-len(s["frames"]))],s["result"].get("foe")) for s in pair])
                proc.stdin.write(both.tobytes())
                if i in (-12,-1): sequence.append(cv2.resize(both,(690,312)))
            sheets.append(np.hstack(sequence))
        proc.stdin.close()
        assert proc.wait()==0
    for i,s in enumerate(sheets):
        cv2.imwrite(str(OUT/f"scene-{i+1}.jpg"),s)
if __name__=="__main__":main()

