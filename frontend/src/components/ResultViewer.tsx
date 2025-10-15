import React, { useEffect, useState } from "react"
import { Spin, Card, Button, Modal } from "antd"
import { useImageNodes } from "../global/useImageNodes"

const { protocol, hostname } = window.location
import { ColumnWidthOutlined } from "@ant-design/icons"
import { ReactCompareSlider } from "react-compare-slider"

const ResultViewer: React.FC<{ imagePath?: string }> = ({ imagePath }) => {
  const { nodes, noseStyle } = useImageNodes()
  const [loading, setLoading] = useState(true)
  const [comparer, setComparer] = useState(false)
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [exists, setExists] = useState(false)

  const imPath = nodes[0] ? nodes[0]?.fullPath : null
  let resultPath = imPath ? imPath.replace("\\FS.png", "\\tuned\\FS.png") : null

  let tuned = ""

  if (typeof imagePath === "string") {
    tuned = resultPath || ""
    resultPath = imagePath
  }

  useEffect(() => {
    if (!resultPath) return

    setImageUrl(null)
    setLoading(true)

    const fetchImage = async () => {
      try {
        const baseUrl = `${protocol}//${hostname}:8001/get-image?im_name=${encodeURIComponent(
          resultPath
        )}`
        const res = await fetch(baseUrl, { cache: "no-store" })
        if (res.ok) {
          setExists(true)
          // 🔥 Add timestamp to bypass browser cache
          setImageUrl(`${baseUrl}&_t=${Date.now()}`)
        } else {
          setExists(false)
        }
      } catch {
        setExists(false)
      } finally {
        setLoading(false)
      }
    }

    fetchImage()
    const interval = setInterval(fetchImage, 1000)
    return () => clearInterval(interval)
  }, [resultPath, noseStyle])

  if (!resultPath)
    return (
      <Card className="w-full h-full flex justify-center items-center text-xl">
        No Input Selected
      </Card>
    )

  if (loading)
    return (
      <div className="flex justify-center items-center w-full h-full">
        <Spin size="large" />
      </div>
    )

  return (
    <div className="flex-1 !aspect-square max-h-[calc(100vh-160px)] relative flex justify-center items-center">
      <Modal
        footer={false}
        width={"auto"}
        open={comparer}
        className="max-w-[90vw]"
        onOk={() => setComparer(false)}
        title="Please select a image as nose style"
        onCancel={() => setComparer(false)}
      >
        <div className="flex gap-10 max-h-[75vh] overflow-hidden justify-center flex-wrap">
          {!imagePath && exists && imageUrl && (
            <ReactCompareSlider
              className="max-h-[70vh] aspect-square"
              itemOne={
                <img
                  alt="Generated Result"
                  src={`${protocol}//${hostname}:8001/get-image?im_name=${encodeURIComponent(
                    nodes[0]?.fullPath || ""
                  )}&_t=${Date.now()}`}
                  className="w-full h-full object-cover rounded"
                />
              }
              itemTwo={
                <img
                  src={`${protocol}//${hostname}:8001/get-image?im_name=${encodeURIComponent(
                    resultPath
                  )}&_t=${Date.now()}`}
                  alt="Generated Result"
                  className="w-full h-full object-cover rounded"
                />
              }
            />
          )}
        </div>
      </Modal>

      {exists && imageUrl ? (
        <>
          {!imagePath && (
            <Button
              type="primary"
              className="absolute top-4 right-4"
              icon={<ColumnWidthOutlined />}
              onClick={() => setComparer(true)}
            />
          )}
          <img
            src={imageUrl}
            alt="Generated Result"
            className="max-h-[calc(100vh-160px)] aspect-square h-full object-cover rounded"
          />
        </>
      ) : (
        <Card className="w-full h-full flex justify-center items-center text-2xl">
          No Results Found
        </Card>
      )}
    </div>
  )
}

export default ResultViewer
