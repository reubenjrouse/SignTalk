let APP_ID = 'd7d8553938894f0893dc8a9890ce92ba'

let token = null;
let uid = String(Math.floor(Math.random()*10000))

let client;
let channel;

let queryString = window.location.search
let urlParams = new URLSearchParams(queryString)
let roomId = urlParams.get('room')

if(!roomId){
    window.location = 'lobby.html'
}

let localStream;
let remoteStream;
let peerConnection;
let localVideoCanvas;
let remoteVideoCanvas;

const servers = {
    iceServers:[
        {
            urls: ['stun:stun1.l.google.com:19302','stun:stun2.l.google.com:19302']
        }
    ]
}
let constraints = {
    video:{
        width:{min: 640, ideal: 1920, max: 1920},
        height: {min:480, ideal:1080, max:1080}
    },
    audio: true
}

let init = async() => {

    client = await AgoraRTM.createInstance(APP_ID)
    await client.login({uid, token})


    channel = client.createChannel(roomId)
    await channel.join()

    channel.on('MemberJoined', handleUserJoined)
    channel.on('MemberLeft', handleUserLeft)

    client.on('MessageFromPeer', handleMessageFromPeer)

    localStream = await navigator.mediaDevices.getUserMedia(constraints)
    document.getElementById('user-1').srcObject = localStream

    localVideoCanvas = document.createElement('canvas');
    remoteVideoCanvas = document.createElement('canvas');
    document.body.appendChild(localVideoCanvas);
    document.body.appendChild(remoteVideoCanvas);

    setInterval(processLocalFrame, 100); 


}
let processLocalFrame = async () => {
    if (!localStream) return;

    const videoTrack = localStream.getVideoTracks()[0];
    const imageCapture = new ImageCapture(videoTrack);
    
    try {
        const frame = await imageCapture.grabFrame();

        localVideoCanvas.width = frame.width;
        localVideoCanvas.height = frame.height;
        const ctx = localVideoCanvas.getContext('2d');
        ctx.drawImage(frame, 0, 0);

        const imageData = localVideoCanvas.toDataURL('image/jpeg');

        // Send frame to backend for processing
        const response = await fetch('http://127.0.0.1:5000/process_frame', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ frame: imageData }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();

        // Display processed frame
        const processedImage = new Image();
        processedImage.onload = () => {
            const ctx = localVideoCanvas.getContext('2d');
            ctx.drawImage(processedImage, 0, 0);
        };
        processedImage.src = result.processed_frame;

        // Use keypoints as needed
        console.log('Keypoints:', result.keypoints);
        // console.log('Prediction:', result.prediction);
    } catch (error) {
        console.error('Error processing frame:', error);
    }
}
let handleUserLeft = (MemberID) => {
    document.getElementById('user-2').style.display = 'none'
    document.getElementById('user-1').classList.remove('smallFrame')
}
let handleMessageFromPeer = async(message, MemberID) => {
    message = JSON.parse(message.text)
    if(message.type === 'offer'){
        createAnswer(MemberID, message.offer)
    }
    if(message.type === 'answer'){
        addAnswer(message.answer)
    }
    if(message.type === 'candidate'){
        if(peerConnection){
            peerConnection.addIceCandidate(message.candidate)
        }
    }
}
let handleUserJoined = async (MemberID) => {
    console.log('A new user joined the channel:', MemberID)
    createOffer(MemberID)
}
let createPeerConnection = async (MemberID) => {
    peerConnection = new RTCPeerConnection(servers)

    remoteStream = new MediaStream()
    document.getElementById('user-2').srcObject = remoteStream
    document.getElementById('user-2').style.display = 'block'

    document.getElementById('user-1').classList.add('smallFrame')
    if(!localStream){
        localStream = await navigator.mediaDevices.getUserMedia({video: true, audio: false})
        document.getElementById('user-1').srcObject = localStream
    }

    localStream.getTracks().forEach((track)=>{
        peerConnection.addTrack(track, localStream)
    })

    peerConnection.ontrack = (event)=> {
        event.streams[0].getTracks().forEach((track) => {
            remoteStream.addTrack(track)
        })
    }
    
    peerConnection.onicecandidate = async (event) => {
        if (event.candidate){
            client.sendMessageToPeer({text:JSON.stringify({'type':'candidate','candidate':event.candidate})}, MemberID)
        }
    }
}
let createOffer = async(MemberID) => {
    await createPeerConnection(MemberID)
    let offer =  await peerConnection.createOffer()
    await peerConnection.setLocalDescription(offer)

    client.sendMessageToPeer({text:JSON.stringify({'type':'offer','offer':offer})}, MemberID)
}
let createAnswer = async (MemberID, offer) => {
    await createPeerConnection(MemberID)

    await peerConnection.setRemoteDescription(offer)

    let answer = await peerConnection.createAnswer()
    await peerConnection.setLocalDescription(answer)

    client.sendMessageToPeer({text:JSON.stringify({'type':'answer','answer':answer})}, MemberID)



}
let addAnswer = async (answer) => {
    if(!peerConnection.currentRemoteDescription){
        peerConnection.setRemoteDescription(answer)
    }

}
let leaveChannel = async () => {
    await channel.leave()
    await client.logout()
}

let toggleCamera = async() => {
    let videoTrack = localStream.getTracks().find(track=> track.kind === 'video')
    if(videoTrack.enabled){
        videoTrack.enabled = false
        document.getElementById('camera-btn').style.backgroundColor = 'rgb(255,80,80)'
    }
    else{
        videoTrack.enabled = true
        document.getElementById('camera-btn').style.backgroundColor = 'rgb(179,102,249)'
    }
}

let toggleMic = async() => {
    let audioTrack = localStream.getTracks().find(track=> track.kind === 'audio')
    if(audioTrack.enabled){
        audioTrack.enabled = false
        document.getElementById('mic-btn').style.backgroundColor = 'rgb(255,80,80)'
    }
    else{
        audioTrack.enabled = true
        document.getElementById('mic-btn').style.backgroundColor = 'rgb(179,102,249)'
    }
}

function drawKeypoints(ctx, keypoints, width, height) {
    const keypointSize = 5; // Size of the keypoints
    const keypointsPerHand = 21 * 3; // 21 landmarks per hand with x, y, z coordinates

    // Normalize keypoints to the canvas size
    const normalize = (value, max) => value * (width / max);

    // Draw keypoints for left and right hands
    for (let i = 0; i < keypoints.length; i += 3) {
        const x = normalize(keypoints[i] * width, 1); // Assuming keypoints[i] is normalized
        const y = normalize(keypoints[i + 1] * height, 1); // Assuming keypoints[i + 1] is normalized
        ctx.beginPath();
        ctx.arc(x, y, keypointSize, 0, 2 * Math.PI);
        ctx.fillStyle = 'red'; // Color of the keypoints
        ctx.fill();
    }
}

window.addEventListener('beforeunload', leaveChannel)
document.getElementById('camera-btn').addEventListener('click', toggleCamera)
document.getElementById('mic-btn').addEventListener('click', toggleMic)


init()