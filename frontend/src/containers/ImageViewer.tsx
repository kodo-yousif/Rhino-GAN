import { Button, notification, Slider, Spin, theme } from "antd"
import { Fragment, useEffect, useRef, useState } from "react"
import axios from "axios"
import classNames from "classnames"
import {
  Landmark,
  RawNode,
  useImageNodes,
  SegmentationData,
} from "../global/useImageNodes"
import SegmentationCanvas from "../components/SegmentationCanvas"

const DOT_SIZE = 12
const LINE_WIDTH = 2

const getDotLineCoordinate = (coordinate: number): string =>
  `${((coordinate + LINE_WIDTH + DOT_SIZE / 2) / 1024) * 100}%`

export default function ImageViewer({
  node,
  landmarks,
  rightButton,
  selectedModel,
  showLandmarks,
  showSegmentation,
  SegmentationData,
  referenceLandmarks,
  draw,
  undraw,
  setLandmarks,
  setSegmentationData,
  setReferenceLandmarks,
}: {
  node?: RawNode
  selectedModel: string
  rightButton?: boolean
  landmarks: Landmark[]
  showLandmarks: boolean
  showSegmentation?: boolean
  referenceLandmarks: Landmark[]
  SegmentationData: SegmentationData
  setLandmarks: (landmarks: Landmark[]) => void
  undraw: (row: number, column: number, paintSize: number) => void
  setReferenceLandmarks: (landmarks: Landmark[]) => void
  draw: (
    row: number,
    column: number,
    newRegion: number,
    paintSize: number
  ) => void
  setSegmentationData: (newSegmentationData: SegmentationData) => void
}) {
  const { noseStyle, modules } = useImageNodes()
  const [isLoading, setIsLoading] = useState(true)
  const [imgUrl, setImgUrl] = useState<string | null>(null)
  const [dragIndex, setDragIndex] = useState<number | null>(null)
  const [activeLandmarkIndex, setActiveLandmarkIndex] = useState<number | null>(
    null
  )

  const [api, contextHolder] = notification.useNotification()

  const containerRef = useRef<HTMLDivElement>(null)

  const {
    token: { borderRadiusLG, colorBgContainer },
  } = theme.useToken()

  const paperStyle = {
    borderRadius: borderRadiusLG,
    background: !isLoading && imgUrl ? "transparent" : colorBgContainer,
  }

  const fetchImage = async (fullPath: string, signal: AbortSignal) => {
    try {
      const { data } = await axios.get("/image", {
        params: { fullPath },
        signal,
      })

      console.log(data)

      setSegmentationData({
        ...data.segmentationData,
        segmentedImageRef: JSON.parse(
          JSON.stringify(data.segmentationData.segmentedImage)
        ),
      })

      setImgUrl(data.image)

      if (!landmarks.length) setLandmarks(data.landmarks)

      if (!referenceLandmarks.length) setReferenceLandmarks(data.landmarks)

      setIsLoading(false)
    } catch (error) {
      if (axios.isCancel(error)) return
      console.error("Error fetching image:", error)
    }
  }

  const toggleActiveLandmark = (landmarkIndex: number) => {
    setActiveLandmarkIndex(
      landmarkIndex === activeLandmarkIndex ? null : landmarkIndex
    )
  }

  useEffect(() => {
    setIsLoading(true)

    const controller = new AbortController()

    if (node) fetchImage(node.fullPath, controller.signal)

    return () => controller.abort()
  }, [node])

  const handleMouseDown = (index: number) => {
    setDragIndex(index)
  }

  const handleMouseUp = () => {
    setDragIndex(null)
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (dragIndex === null || !containerRef.current) return

    const rect = containerRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left - DOT_SIZE / 2
    const y = e.clientY - rect.top - DOT_SIZE / 2

    const clampedX = Math.max(0, Math.min(x, rect.width))
    const clampedY = Math.max(0, Math.min(y, rect.height))

    const normX = (clampedX / rect.width) * 1024
    const normY = (clampedY / rect.height) * 1024

    setLandmarks(
      landmarks.map((point, i) =>
        i === dragIndex ? [normX, normY, point[2]] : point
      )
    )
  }

  const invertImage = async () => {
    const payload = {
      video: true,
      W_steps: 1100,
      FS_steps: 250,
      im: node?.fullPath,
    }

    try {
      const {
        data: { message },
      } = await axios.post("/invert", payload)

      api.success({
        message: "Inversion",
        description: message,
        placement: "bottomRight",
      })
    } catch (error: any) {
      console.log(error)
      api.error({
        message: "Inversion",
        placement: "bottomRight",
        description: (
          <div>
            <div>Inversion Failed:</div>
            <div className="text-red-500">
              {error?.response?.data || error?.message}
            </div>
          </div>
        ),
      })
    }
  }

  const styleImage = async () => {
    const changedLandmarks = landmarks.map((landmark, idx) => {
      return [
        ...landmark,
        (idx > 26 && idx < 36) ||
        landmark.join(",") !== referenceLandmarks[idx].join(",")
          ? // true
            1
          : 0,
      ]
    })

    const payload: Partial<Record<keyof typeof modules | "model", any>> = {
      model: selectedModel,
    }

    try {
      if (modules.segmentation)
        payload.segmentation = SegmentationData?.segmentedImage

      if (modules.noseStyle) payload.noseStyle = noseStyle
      if (modules.landmarks) payload.landmarks = changedLandmarks
      if (modules?.doStyle) payload.doStyle = modules.doStyle

      const {
        data: { message },
      } = await axios.post("/fine-tune", {
        fullPath: node?.fullPath,
        ...payload,
      })
      api.success({
        message: "Tunning",
        description: message,
        placement: "bottomRight",
      })
    } catch (error: any) {
      console.log(error)
      api.error({
        message: "Tunning",
        placement: "bottomRight",
        description: (
          <div>
            <div>Tunning Failed:</div>
            <div className="text-red-500">
              {error?.response?.data || error?.message}
            </div>
          </div>
        ),
      })
    }
  }

  if (!node)
    return (
      <div className="aspect-square flex items-center justify-center">
        Please select an image
      </div>
    )

  let showSlider = true

  let sliderValue = 0

  try {
    if (activeLandmarkIndex !== null && showLandmarks) {
      sliderValue = landmarks[activeLandmarkIndex][2]
    }
  } catch (error) {
    sliderValue = 0
    showSlider = false
  }

  return (
    <div className="relative aspect-square">
      {contextHolder}
      <div
        className={classNames("absolute w-git flex top-4 z-40 ", {
          "left-20": !rightButton,
          "right-20": rightButton,
        })}
      >
        {node?.fullPath.includes("output") ? (
          <Button type="primary" onClick={styleImage}>
            Style
          </Button>
        ) : (
          <Button type="primary" onClick={invertImage}>
            Invert
          </Button>
        )}
      </div>
      <div
        style={paperStyle}
        className={classNames(
          "overflow-hidden h-[calc(100vh-160px)] aspect-square",
          {
            "w-full": !(!isLoading && imgUrl),
          }
        )}
      >
        {!isLoading && imgUrl ? (
          <div
            ref={containerRef}
            onMouseUp={handleMouseUp}
            onMouseMove={handleMouseMove}
            className="aspect-square relative w-full max-h-[100%] max-w-[100%] object-cover"
          >
            <div className="relative aspect-square w-full max-h-[100%] max-w-[100%] size-fit">
              {showSegmentation && SegmentationData && (
                <SegmentationCanvas
                  draw={draw}
                  undraw={undraw}
                  SegmentationData={SegmentationData}
                />
              )}
              {/* {showSegmentation && SegmentationData && (
                <div className="absolute flex items-stretch flex-col w-full h-full left-0 top-0">
                  {SegmentationData.segmentedImage.map((row, rowIndex) => (
                    <div key={rowIndex} className="flex-1 flex items-stretch">
                      {row.map((column, columnIndex) => (
                        <div
                          key={`${rowIndex}-${columnIndex}`}
                          // @ts-ignore
                          data={column.toString()}
                          className="flex-1 shrink-0 hover:bg-red-500"
                          style={{
                            backgroundColor: `rgb(${SegmentationData.COLOR_MAP[
                              column
                            ].join(",")})`,
                          }}
                        />
                      ))}
                    </div>
                  ))}
                </div>
              )} */}
              <img
                src={imgUrl}
                alt="Loaded"
                style={paperStyle}
                className="aspect-square w-full max-h-[100%] max-w-[100%] object-cover"
              />
            </div>

            {showLandmarks &&
              landmarks.map(
                (landmark, index) =>
                  index >= 27 &&
                  index <= 35 && (
                    <div
                      key={index}
                      onMouseDown={() => handleMouseDown(index)}
                      onClick={() => toggleActiveLandmark(index)}
                      className={classNames(
                        "absolute z-10 cursor-grab drop-shadow-[0_0_4px_rgba(0,0,0,0.8)] bg-red-500 border-2 border-white border-solid -translate-x-1/2 -translate-y-1/2 rounded-full",
                        { "!bg-blue-500": activeLandmarkIndex === index }
                      )}
                      style={{
                        width: DOT_SIZE,
                        height: DOT_SIZE,
                        top: `${(landmark[1] / 1024) * 100}%`,
                        left: `${(landmark[0] / 1024) * 100}%`,
                      }}
                    />
                  )
              )}

            {showLandmarks &&
              referenceLandmarks.map((landmark, index) => {
                const hasMoved = !(
                  landmark[0] === landmarks[index][0] &&
                  landmark[1] === landmarks[index][1]
                )

                return (
                  <Fragment key={index + "references"}>
                    {hasMoved && (
                      <>
                        <svg className="absolute pointer-events-none w-full h-full left-0 top-0">
                          <defs>
                            <linearGradient
                              x1="0%"
                              y1="0%"
                              y2="0%"
                              x2="100%"
                              id="lineGradient"
                            >
                              <stop offset="0%" stopColor="rgb(69, 10, 10)" />
                              <stop
                                offset="100%"
                                stopColor="rgb(239, 68, 68)"
                              />
                            </linearGradient>
                          </defs>

                          <line
                            stroke="url(#lineGradient)"
                            strokeWidth={LINE_WIDTH}
                            x1={getDotLineCoordinate(landmark[0])}
                            y1={getDotLineCoordinate(landmark[1])}
                            x2={getDotLineCoordinate(landmarks[index][0])}
                            y2={getDotLineCoordinate(landmarks[index][1])}
                          />
                        </svg>

                        <div
                          className="absolute z-10 pointer-events-none drop-shadow-[0_0_4px_rgba(0,0,0,0.8)] bg-red-950 border-2 border-slate-400 border-solid -translate-x-1/2 -translate-y-1/2 rounded-full"
                          style={{
                            width: DOT_SIZE,
                            height: DOT_SIZE,
                            top: `${(landmark[1] / 1024) * 100}%`,
                            left: `${(landmark[0] / 1024) * 100}%`,
                          }}
                        />
                      </>
                    )}
                  </Fragment>
                )
              })}
          </div>
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <Spin size="large" />
          </div>
        )}
      </div>
    </div>
  )
}
