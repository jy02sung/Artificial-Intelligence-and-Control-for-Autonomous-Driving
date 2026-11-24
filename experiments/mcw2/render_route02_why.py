"""Annotate the actual route-02 replay with its verified lane topology and policy input test."""
import json
import subprocess
import cv2
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont
from .find_gap_example import OUT

def main():
    data=json.loads((OUT/"example.json").read_bytes())
    event=data["event"]
    probe=next(r for r in json.loads((OUT/"route02-action-probe.json").read_bytes()) if r["step"]==event["change"])
    assert data["row"]["route"]=="route_02" and data["frames"][0]["targets"]==[1,2,3,4]
    cap=cv2.VideoCapture(str(OUT/"ppo-gap-opening.mp4"))
    target=OUT/"route02-why-left.mp4"
    proc=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),"-loglevel","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s","1280x720","-r","10","-i","-","-c:v","libx264","-pix_fmt","yuv420p","-movflags","+faststart",str(target)],stdin=subprocess.PIPE)
    fonts={s:ImageFont.truetype("C:/Windows/Fonts/malgun.ttf",s) for s in (19,22,25,31,35)}
    cyan="#48d8ef";green="#9be2b1";orange="#ffb16a";white="#f2f5f7"
    for i,f in enumerate(data["frames"]):
        ok,raw=cap.read()
        assert ok
        for _ in range(3):cap.grab()
        img=Image.new("RGB",(1280,720),"#16212b")
        dr=ImageDraw.Draw(img)
        def label(x,y,s,size=22,col=white):dr.text((x,y),s,font=fonts[size],fill=col)
        label(35,13,"2번 경로   |   오른쪽 끝 차로가 끊기는 구간",35)
        label(36,76,"실제 주행   ·   하늘색 AV / 주황색 옆 차",22)
        label(568,76,"차로 연결도   ·   진행 방향 ↑",22)
        img.paste(Image.fromarray(cv2.cvtColor(raw[:,:500],cv2.COLOR_BGR2RGB)),(32,119))
        t=f["step"];av=f["av"]
        for lane in range(5):
            x=828-44*lane
            dr.line((x,485,x,275 if lane==0 else 153),fill=orange if lane==0 else green,width=8)
            if lane:
                dr.polygon([(x,140),(x-8,157),(x+8,157)],fill=green)
            label(x-8,491,str(lane),22)
        dr.line((814,274,842,274),fill=orange,width=5)
        label(866,148,"같은 다음 도로로 계속",25,green)
        label(866,194,"1~4번 차로 연결",22)
        label(866,254,"0번 차로 끝",25,orange)
        label(866,296,"왼쪽 출구 분기 없음",22)
        y=274+int(av["remaining"]*1.48)
        x=828-44*av["lane"]
        dr.ellipse((x-9,y-9,x+9,y+9),fill=cyan,outline=white,width=2)
        label(865,375,"도로 경계까지 약 %.0fm"%av["remaining"],22)
        label(865,415,"AV 속도 %.1f km/h"%(3.6*av["speed"]),22,cyan)
        peer=next(p for p in f["near"] if p["id"]==event["foe"])
        gap=peer["pos"]-peer["length"]-av["pos"]
        label(568,549,"옆 차 뒤로 들어갈 범퍼 여유   %+.1f m"%gap,25)
        dr.rectangle((0,632,1280,720),fill="#223444")
        if t<event["change"]:
            label(35,648,"0번 차로에서 감속하며 옆 차가 지나갈 공간을 확보",31)
        else:
            label(35,644,"변경 순간   |   실제 연결 정보 → 왼쪽 변경",25)
            label(35,682,"같은 관측에서 현재 차로도 계속 이어진다고 입력 → 차로 유지",22)
        if t==event["change"]:
            assert probe["original"][1]==1 and probe["current_as_left"][1]==0
            img.save(OUT/"route02-why-left.jpg")
        copies=24 if i==0 else 30 if t==event["change"] else 4
        for _ in range(copies):proc.stdin.write(img.tobytes())
    cap.release();proc.stdin.close();assert proc.wait()==0
if __name__=="__main__":main()
