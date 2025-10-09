import { useCallback, useEffect, useRef, useState } from "react"
import {
  IMAGE_COLS,
  IMAGE_ROWS,
  SegmentationData,
} from "../global/useImageNodes"
import { Input, Switch } from "antd"

interface ISegmentationCanvas {
  SegmentationData: SegmentationData
  undraw: (row: number, column: number, paintSize: number) => void
  draw: (
    row: number,
    column: number,
    newRegion: number,
    paintSize: number
  ) => void
}
export default function SegmentationCanvas({
  draw,
  undraw,
  SegmentationData,
}: ISegmentationCanvas) {
  const [paintSize, setPaintSize] = useState(5)
  const [isPainting, setIsPainting] = useState(false)
  const [isDrawing, setIsDrawing] = useState(true)

  const canvasRef = useRef<HTMLCanvasElement>(null)

  const onDrawingStart = () => setIsPainting(true)
  const onDrawingStop = () => setIsPainting(false)

  const getCellFromEvent = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      if (!isPainting) return

      const canvas = canvasRef.current!
      const rect = canvas.getBoundingClientRect()

      // Map client coords -> canvas pixel coords (handles CSS scaling & DPR)
      const scaleX = canvas.width / rect.width
      const scaleY = canvas.height / rect.height

      const x = (e.clientX - rect.left) * scaleX
      const y = (e.clientY - rect.top) * scaleY

      // Size of one cell in canvas pixels
      const cellW = canvas.width / IMAGE_COLS
      const cellH = canvas.height / IMAGE_ROWS

      // Compute col/row and clamp to bounds
      let col = Math.floor(x / cellW)
      let row = Math.floor(y / cellH)
      col = Math.max(0, Math.min(IMAGE_COLS - 1, col))
      row = Math.max(0, Math.min(IMAGE_ROWS - 1, row))

      if (isDrawing) draw(row, col, SegmentationData!.regions.nose, paintSize)
      else undraw(row, col, paintSize)

      return { row, col }
    },
    [isPainting, isDrawing]
  )

  useEffect(() => {
    if (!SegmentationData || !canvasRef.current) return

    const canvas = canvasRef.current as HTMLCanvasElement
    const ctx = canvas.getContext("2d")

    if (!ctx) return

    const imageData = ctx.createImageData(512, 512)

    for (let row = 0; row < IMAGE_ROWS; row++) {
      for (let col = 0; col < IMAGE_COLS; col++) {
        const index = (row * IMAGE_ROWS + col) * 4
        const label = SegmentationData.segmentedImage[row][col]
        const refLabel = SegmentationData.segmentedImageRef[row][col]

        SegmentationData.segmentedImageRef[row][col]
        let r, g, b

        const isNose = label === SegmentationData.regions.nose
        const isRefNose = refLabel === SegmentationData.regions.nose

        if (isRefNose && !isNose) [r, g, b] = [0, 0, 0]
        else if (!isRefNose && isNose) [r, g, b] = [0, 0, 255]
        else [r, g, b] = SegmentationData.COLOR_MAP[label]

        // if (isNose) [r, g, b] = SegmentationData.COLOR_MAP[label]
        // else [r, g, b] = SegmentationData.COLOR_MAP[label]

        imageData.data[index] = r
        imageData.data[index + 1] = g
        imageData.data[index + 2] = b
        imageData.data[index + 3] = isNose || isRefNose ? 100 : 0
      }
    }

    ctx.putImageData(imageData, 0, 0)
  }, [SegmentationData, isPainting])

  return (
    <>
      <div className="absolute flex items-center gap-2 left-6 bottom-6 z-50">
        <Input
          min={0}
          type="number"
          value={paintSize}
          className="w-[70px]"
          onChange={(e) => setPaintSize(+e.target.value)}
        />
        <Switch
          checked={isDrawing}
          className="bg-red-400"
          onChange={() => setIsDrawing((d) => !d)}
        />
      </div>
      <canvas
        width={512}
        height={512}
        ref={canvasRef}
        onPointerUp={onDrawingStop}
        onPointerDown={onDrawingStart}
        onPointerMove={getCellFromEvent}
        className="absolute left-0 top-0 w-full h-full"
      />
    </>
  )
}
