"""Render measured gap, PPO commands and vehicles from the verified SUMO replay."""
import json
import math
import sys
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

from .find_gap_example import OUT,measure
from .render_scratch_failures import LANES

def main():
    z=json.loads((OUT/(sys.argv[1] if len(sys.argv)>1 else "example.json")).read_bytes())
    ev=z["event"];row=z["row"];side=ev["side"];foe=ev["foe"]
    frames=z["frames"]
    def font(size):return ImageFont.truetype("C:/Windows/Fonts/malgun.ttf",size)
    small,normal,big=font(21),font(28),font(35)
    stem="ppo-gap-opening" if len(sys.argv)<2 else sys.argv[1].split(".")[0]
    movie=OUT/(stem+".mp4")
    proc=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),"-loglevel","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s","1000x500","-r","10","-i","-","-c:v","libx264","-pix_fmt","yuv420p","-movflags","+faststart",str(movie)],stdin=subprocess.PIPE)
    for i,f in enumerate(frames):
        av=f["av"];center=np.array(av["xy"]);theta=math.radians(av["angle"])
        mat=7*np.array([[math.cos(theta),-math.sin(theta)],[-math.sin(theta),-math.cos(theta)]])
        def pix(pts):return ((pts-center)@mat.T+[245,300]).astype(np.int32)
        road=np.full((500,500,3),[27,32,39],np.uint8)
        for pts,lo,hi in LANES:
            if np.all(lo<center+75) and np.all(hi>center-75):
                line=pix(pts)
                cv2.polylines(road,[line],False,(70,76,84),23,cv2.LINE_AA)
                cv2.polylines(road,[line],False,(99,106,113),1,cv2.LINE_AA)
        for car in f["near"]+[av]:
            ang=math.radians(car["angle"]);tip=np.array(car["xy"]);back=car["length"]*np.array([math.sin(ang),math.cos(ang)])
            s=.5*car["width"]*np.array([math.cos(ang),-math.sin(ang)])
            color=(59,212,242) if car is av else (255,159,66) if car["id"]==foe else (170,180,185)
            cv2.fillConvexPoly(road,pix(np.array([tip+s,tip-s,tip-back-s,tip-back+s])),color,cv2.LINE_AA)
        img=Image.new("RGB",(1000,500),(24,29,35));img.paste(Image.fromarray(road));dr=ImageDraw.Draw(img)
        def txt(x,y,t,face=normal,color="white"):dr.text((x,y),t,font=face,fill=color)
        peer=next((p for p in f["near"] if p["id"]==foe),None)
        gap=measure(av,peer)[side] if peer and av["edge"]==peer["edge"] else None
        t=f["step"]
        label=("합류 차로 주행" if av["lane"]==ev["lane"] else "원래 차로 복귀") if t>ev["change"] else "차로변경 명령" if t==ev["change"] else ("상대 차량을 먼저 보냄" if side==0 else "상대 차량 앞으로 이동")
        txt(520,16,f'경로 {row["route"][-2:]}  ·  PPO 15만 스텝',small)
        txt(520,65,label,big)
        txt(520,132,f'AV 속도    {av["speed"]*3.6:.1f} km/h',normal,(59,212,242))
        if peer:txt(520,176,f'상대 속도    {peer["speed"]*3.6:.1f} km/h',normal,(255,159,66))
        which="뒤" if side==0 else "앞"
        txt(520,240,f'상대 차 {which}로 들어갈 여유',small)
        txt(520,273,f'{gap:+.1f} m' if gap is not None else "다른 도로 구간",big)
        txt(520,339,f'가감속 명령 {f["accel"]:+.2f}   |   차로 {av["lane"]}',small)
        txt(520,389,f'변경 명령 기준 {(t-ev["change"])*.2:+.1f}초',normal)
        txt(520,452,"양수 = 선택한 쪽으로 범퍼 사이 공간 확보",small)
        if t==ev["change"]:img.save(OUT/(stem+".jpg"))
        for _ in range(4):proc.stdin.write(img.tobytes())
    proc.stdin.close()
    assert proc.wait()==0
if __name__=="__main__":main()


