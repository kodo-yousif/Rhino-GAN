import { TreeDataNode } from "antd"

import { FolderOutlined, FileImageOutlined } from "@ant-design/icons"
import {
  RawNode,
  SegmentationData,
  useImageNodes,
} from "../global/useImageNodes"
import ImageViewer from "../containers/ImageViewer"
import ResultViewer from "../components/ResultViewer"

const addIcons = (nodes: RawNode[]): TreeDataNode[] => {
  return nodes.map((node) => ({
    ...node,
    icon: node.children ? <FolderOutlined /> : <FileImageOutlined />,
    children: node.children ? addIcons(node.children) : undefined,
  }))
}

export default function Home() {
  const {
    nodes,
    noseStyle,
    showSegmentation,
    segmentationDatas,
    nodeLandmarksInfo,
    showNodeLandmarks,
    drawSegmentationByIndex,
    undrawSegmentationByIndex,
    setSegmentationDataByIndex,
    setNodeLandmarksInfoByIndex,
  } = useImageNodes()

  const selectedNode = nodes[0]

  return (
    <div className="w-full h-[calc(100vh-150px)] overflow-hidden flex gap-4 items-center justify-center">
      {noseStyle === "self" || noseStyle === selectedNode?.fullPath ? (
        <div className="flex-1 aspect-square flex justify-center items-center">
          <ImageViewer
            selectedModel="f"
            node={nodes[0]}
            showSegmentation={showSegmentation[0]}
            SegmentationData={segmentationDatas[0]}
            setSegmentationData={(newSegmentationData: SegmentationData) =>
              setSegmentationDataByIndex(0, newSegmentationData)
            }
            draw={(row, column, newRegion, paintSize) =>
              drawSegmentationByIndex(0, row, column, newRegion, paintSize)
            }
            undraw={(row, column, paintSize) =>
              undrawSegmentationByIndex(0, row, column, paintSize)
            }
            {...nodeLandmarksInfo[0]}
            showLandmarks={showNodeLandmarks[0]}
            setLandmarks={(landmarks) =>
              setNodeLandmarksInfoByIndex(0, "landmarks", landmarks)
            }
            setReferenceLandmarks={(landmarks) =>
              setNodeLandmarksInfoByIndex(0, "referenceLandmarks", landmarks)
            }
          />
        </div>
      ) : (
        <ResultViewer imagePath={selectedNode?.fullPath} />
      )}
      <ResultViewer />
    </div>
  )
}
