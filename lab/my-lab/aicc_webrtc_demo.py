"""my-lab WebRTC DataChannel 루프백 데모 — 별도 프로세스(신선한 이벤트 루프)로 실행.

이유: Python 3.14 + aiortc 1.15 는 `asyncio.wait_for` 를 task 안에서만 허용하는데,
aiortc 내부(dtls transport)가 이를 사용해 노트북의 실행 중인 루프(nest_asyncio)에서 깨진다.
→ subprocess 로 띄워 깨끗한 asyncio.run() 환경에서 동작시킨다. (노트 07 [REAL] 셀에서 호출)
"""
import asyncio
import json
import time

from aiortc import RTCPeerConnection


async def datachannel_loopback():
    pc1, pc2 = RTCPeerConnection(), RTCPeerConnection()
    ch = pc1.createDataChannel("aicc-meta")
    result, done = {}, asyncio.Event()

    @pc2.on("datachannel")
    def on_channel(channel):
        @channel.on("message")
        def on_message(msg):
            channel.send(msg)                      # pc2는 받은 즉시 에코

    @ch.on("open")
    def on_open():
        result["t0"] = time.perf_counter()
        ch.send(json.dumps({"type": "chunk", "seq": 0, "text": "ping", "dur_ms": 0}))

    @ch.on("message")
    def on_echo(msg):
        result["rtt_ms"] = (time.perf_counter() - result["t0"]) * 1000
        result["echo"] = json.loads(msg)
        done.set()

    # 수동 시그널링: 오퍼/앤서를 변수로 직접 전달 (실서비스에선 이 구간이 시그널링 서버)
    await pc1.setLocalDescription(await pc1.createOffer())
    await pc2.setRemoteDescription(pc1.localDescription)
    await pc2.setLocalDescription(await pc2.createAnswer())
    await pc1.setRemoteDescription(pc2.localDescription)
    sdp_head = [l for l in pc1.localDescription.sdp.split("\n")
                if l.startswith(("m=", "a=ice-ufrag", "a=fingerprint"))]
    t0 = time.perf_counter()
    while not done.is_set():
        if time.perf_counter() - t0 > 10:
            raise TimeoutError("DataChannel 에코 타임아웃")
        await asyncio.sleep(0.05)
    await pc1.close()
    await pc2.close()
    return result, sdp_head


if __name__ == "__main__":
    res, sdp = asyncio.run(datachannel_loopback())
    print("###SDP###")
    for line in sdp[:3]:
        print(line.strip())
    print("###RESULT###")
    print(json.dumps(res, ensure_ascii=False))
