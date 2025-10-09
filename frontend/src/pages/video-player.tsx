import { useEffect, useRef, useState } from "react"

export default function VideoPlayer() {
  const [videoUrl, setVideoUrl] = useState("")

  const videoRef = useRef<HTMLVideoElement | null>(null)

  useEffect(() => {
    const filePath =
      "C:\\projects\\Research\\nose\\images\\outputs\\5\\video\\FS_optimization.mp4"
    const encodedPath = encodeURIComponent(filePath.replace(/\\/g, "/"))

    fetch(
      `http://desktop-sgilgur.tail6aeec9.ts.net:3001/video?fullPath=${encodedPath}`
    )
      .then((res) => res.arrayBuffer())
      .then((buffer) => {
        const blob = new Blob([buffer], { type: "video/mp4" })
        const url = URL.createObjectURL(blob)
        setVideoUrl(url)
      })
      .catch((err) => {
        console.error("Failed to load video:", err)
      })
  }, [])

  return (
    <video ref={videoRef} controls width="720" height="400" src={videoUrl} />
  )
}
