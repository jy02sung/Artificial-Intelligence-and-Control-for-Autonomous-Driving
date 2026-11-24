"""Render a route overview and the continuous before/after traffic scene in Korean."""
import json
import math
import subprocess
import sys
import cv2
import numpy as np
import imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
from .long_gap_replays import OUT
from .find_gap_example import measure
from .render_scratch_failures import LANES,ROOT

def render(path):
    scene=json.loads(path.read_bytes());e=scene["event"];row=scene["row"];frames=scene["frames"]
    model=int(scene["model"])//10000;target=e["lane"];direction=scene["direction"]
    change=e["change"];subject=e["foe"];side=e["side"]
    font={s:ImageFont.truetype("C:/Windows/Fonts/malgun.ttf",s) for s in (18,21,23,26,31,37)}
    source=next(f for f in frames if f["step"]==change)
    origin=source["av"]["lane"];targets=source["targets"]
    proc=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),"-loglevel","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s","1100x620","-r","10","-i","-","-c:v","libx264","-preset","veryfast","-pix_fmt","yuv420p","-movflags","+faststart",str(path.with_suffix(".mp4"))],stdin=subprocess.PIPE)
    for idx,f in enumerate(frames):
        av=f["av"];t=f["step"];pos=np.array(av["xy"]);angle=math.radians(av["angle"])
        mat=5.1*np.array([[math.cos(angle),-math.sin(angle)],[-math.sin(angle),-math.cos(angle)]])
        def px(points):return ((points-pos)@mat.T+[270,300]).astype(np.int32)
        road=np.full((484,550,3),[25,32,37],np.uint8)
        for pts,lo,hi in LANES:
            if np.all(lo<pos+85) and np.all(hi>pos-85):
                line=px(pts);cv2.polylines(road,[line],False,(72,77,82),18,cv2.LINE_AA)
                cv2.polylines(road,[line],False,(101,108,114),1,cv2.LINE_AA)
        for car in f["near"]+[av]:
            a=math.radians(car["angle"]);tip=np.array(car["xy"]);back=car["length"]*np.array([math.sin(a),math.cos(a)])
            s=.5*car["width"]*np.array([math.cos(a),-math.sin(a)])
            color=(57,218,241) if car is av else (255,158,57) if car["id"]==subject else (172,180,185)
            cv2.fillConvexPoly(road,px(np.array([tip+s,tip-s,tip-back-s,tip-back+s])),color,cv2.LINE_AA)
        img=Image.new("RGB",(1100,620),"#15212b");img.paste(Image.fromarray(road),(15,79))
        dr=ImageDraw.Draw(img)
        def txt(x,y,value,size=23,color="white"):dr.text((x,y),value,font=font[size],fill=color)
        txt(22,12,f"경로 {row['route'][-2:]}   ·   PPO {model}만 스텝",37)
        txt(584,24,f"차로변경 명령 기준   {(t-change)*.2:+.1f}초",26)
        txt(584,86,"왜 옆 차로로 이동했나",21,"#95daba")
        txt(584,122,"다음 도로로 가려면 차로 변경 필요",26)
        txt(584,166,f"변경 전 {origin}번   →   진입 {target}번  ({'왼쪽' if direction==1 else '오른쪽'})",23)
        txt(584,204,"경로 연결 차로   "+", ".join(map(str,targets))+"번",21)
        txt(584,247,"주행 상황",21,"#95daba")
        phase=("변경 전 접근" if t<e["start"] else ("감속하며 상대를 먼저 보냄" if side==0 else "가속하며 상대 앞으로 이동") if t<change else "차로변경 명령" if t==change else "변경 이후 주행")
        txt(584,280,phase,31)
        txt(584,335,f"AV   {av['speed']*3.6:.1f} km/h",26,"#39daf1")
        peer=next((p for p in f["near"] if p["id"]==subject),None)
        if peer:txt(584,374,f"상대 차량   {peer['speed']*3.6:.1f} km/h",23,"#ff9e39")
        gap=measure(av,peer)[side] if peer and av["edge"]==peer["edge"] else None
        label='뒤로' if side==0 else '앞으로'
        txt(584,430,f"상대 {label} 들어갈 범퍼 여유",21)
        txt(584,464,f"{gap:+.1f} m" if gap is not None else "같은 도로 측정 범위 밖",26)
        txt(584,517,f"가감속 명령 {f['accel']:+.2f}    현재 차로 {av['lane']}",21)
        txt(23,578,"하늘색  AV     주황색  추적 상대     회색  주변 차량",21)
        txt(588,578,"실제 SUMO 재생  ·  접근~변경 후 0.5배속",21)
        if t==change:img.save(path.with_suffix(".jpg"))
        repeat=16 if idx==0 or t==change else 4
        for _ in range(repeat):proc.stdin.write(img.tobytes())
    proc.stdin.close();assert proc.wait()==0
if __name__=="__main__":
    for name in sys.argv[1:]:render(OUT/(name+".json"))
