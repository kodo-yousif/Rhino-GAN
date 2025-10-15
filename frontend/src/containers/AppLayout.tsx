import { Outlet } from "react-router-dom"

import { theme, Button, Layout, Modal, Switch, notification } from "antd"

import { GlobalLoading } from "@/components/GlobalLoading"
import { useImageNodes } from "../global/useImageNodes"
import { useCallback, useEffect, useState } from "react"
import axios from "axios"
import classNames from "classnames"
import { ReloadOutlined } from "@ant-design/icons"
import UploadImage from "../components/UploadImage"
import ServerStatus from "../components/ServerStatus"
const { Header, Content } = Layout

const { protocol, hostname } = window.location

export default function AppLayout() {
  const [images, setImages] = useState<string[]>([])
  const [selectImageModal, setSelectImageModal] = useState(false)
  const [selectStyleModal, setSelectStyleModal] = useState(false)
  const [api, contextHolder] = notification.useNotification()

  const {
    token: { colorBgContainer },
  } = theme.useToken()

  const {
    nodes,
    noseStyle,
    showSegmentation,
    showNodeLandmarks,
    setNoseStyle,
    setNodeByIndex,
    toggleLandmarkByIndex,
    toggleSegmentationByIndex,
  } = useImageNodes()

  const selectedNode = nodes[0]

  const setSelectedNode = (fullPath: string) => {
    setNodeByIndex(
      0,
      fullPath === selectedNode?.fullPath
        ? undefined
        : {
            fullPath,
            key: fullPath,
            title: fullPath,
            children: [],
            type: "image",
          }
    )

    setSelectImageModal(false)
  }

  const setNewStyle = (fullPath: string) => {
    setNoseStyle(!!fullPath.trim().length ? fullPath.trim() : "self")
    setSelectStyleModal(false)
  }

  const fetchTreeData = useCallback(async (signal?: AbortSignal) => {
    try {
      const response = await axios.get("/get-images", {
        signal,
      })

      setImages(response.data)
    } catch (error) {
      console.log(error)

      setImages([])
    }
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    fetchTreeData(controller.signal)
    return () => controller.abort()
  }, [fetchTreeData])

  const transferImage = async () => {
    try {
      const {
        data: { message },
      } = await axios.post("/fine-tune", {
        fullPath: selectedNode?.fullPath,
        noseStyle,
      })

      api.success({
        message: "Transferring",
        description: message,
        placement: "bottomRight",
      })
    } catch (error: any) {
      console.log(error)
      api.error({
        message: "Transferring",
        placement: "bottomRight",
        description: <div>Tunning Failed</div>,
      })
    }
  }

  return (
    <Layout className="h-screen overflow-hidden">
      {contextHolder}
      <GlobalLoading />
      <Layout>
        <Header
          style={{ background: colorBgContainer }}
          className="p-4 h-[100px] flex items-center justify-between"
        >
          <div className="flex gap-6 h-full items-center">
            <button
              className="size-[68px] cursor-pointer hover:shadow border-none flex justify-center items-center aspect-square rounded bg-transparent"
              onClick={() => setSelectImageModal(true)}
            >
              {!selectedNode ? (
                <div>Select a image</div>
              ) : (
                <img
                  className="size-[68px] rounded"
                  src={`${protocol}//${hostname}:8001/get-image?im_name=${selectedNode.fullPath}`}
                />
              )}
            </button>

            <button
              className="size-[68px] cursor-pointer hover:shadow border-none flex justify-center items-center aspect-square rounded bg-transparent"
              onClick={() => setSelectStyleModal(true)}
            >
              {noseStyle === "self" || noseStyle === selectedNode?.fullPath ? (
                <div className="text-blue-500 font-semibold">Self Style</div>
              ) : (
                <img
                  className="size-[68px] rounded"
                  src={`${protocol}//${hostname}:8001/get-image?im_name=${noseStyle}`}
                />
              )}
            </button>
            {noseStyle === "self" || noseStyle === selectedNode?.fullPath ? (
              <div className="flex gap-6 h-full">
                <div className="flex items-center justify-center -mt-5 ms-10 flex-col">
                  <div className="h-[50px]">Show Landmark:</div>
                  <Switch
                    size="small"
                    checked={showNodeLandmarks[0]}
                    onChange={() => toggleLandmarkByIndex(0)}
                  />
                </div>

                <div className="flex items-center justify-center flex-col -mt-5">
                  <div className="h-[50px]">Show Segmentation:</div>
                  <Switch
                    size="small"
                    checked={showSegmentation[0]}
                    onChange={() => toggleSegmentationByIndex(0)}
                  />
                </div>
              </div>
            ) : (
              <Button type="primary" onClick={transferImage}>
                Transfer
              </Button>
            )}
          </div>

          <div className="flex flex-row-reverse items-center gap-8">
            {/* <DarkModeSwitch
              size={25}
              checked={!isDark}
              onChange={toggleTheme}
              sunColor={colorPrimary}
              moonColor={colorPrimary}
            /> */}

            <ServerStatus />
            <UploadImage />

            <Button
              type="primary"
              icon={<ReloadOutlined />}
              onClick={() => fetchTreeData()}
            />

            {/* <Button
              type="primary"
              onClick={() => {
                const { protocol, hostname } = window.location
                const swaggerApi = `${protocol}//${hostname}:38001/docs`
                window.open(swaggerApi, "_blank")
              }}
            >
              Swagger
            </Button> */}
          </div>
        </Header>
        <Content className="m-4 flex flex-col">
          <div className="h-full w-full overflow-auto" style={{}}>
            <Modal
              footer={false}
              width={"auto"}
              open={selectImageModal}
              className="max-w-[90vw]"
              title="Please select a image"
              onOk={() => setSelectImageModal(false)}
              onCancel={() => setSelectImageModal(false)}
            >
              <div className="flex gap-10 max-h-[75vh] overflow-auto justify-center flex-wrap">
                {!!images.length &&
                  images.map((im) => (
                    <img
                      key={im}
                      onClick={() => setSelectedNode(im)}
                      className={classNames(
                        "transition-all cursor-pointer duration-300 rounded hover:rounded-[30px] hover:scale-125 size-[200px]",
                        {
                          "!rounded-full": im === selectedNode?.fullPath,
                        }
                      )}
                      src={`${protocol}//${hostname}:8001/get-image?im_name=${encodeURIComponent(
                        im
                      )}`}
                    />
                  ))}
              </div>
            </Modal>

            <Modal
              footer={false}
              width={"auto"}
              open={selectStyleModal}
              className="max-w-[90vw]"
              onOk={() => setSelectStyleModal(false)}
              title="Please select a image as nose style"
              onCancel={() => setSelectStyleModal(false)}
            >
              <div className="flex gap-10 max-h-[75vh] overflow-auto justify-center flex-wrap">
                <button
                  onClick={() => setNewStyle("self")}
                  className={classNames(
                    "transition-all text-lg border-none cursor-pointer bg-slate-300 duration-300 rounded hover:rounded-[30px] hover:scale-125 size-[200px]",
                    {
                      "!rounded-full": "self" === noseStyle,
                    }
                  )}
                >
                  Self Style
                </button>
                {!!images.length &&
                  images.map((im) => (
                    <img
                      key={im}
                      onClick={() => setNewStyle(im)}
                      className={classNames(
                        "transition-all cursor-pointer duration-300 rounded hover:rounded-[30px] hover:scale-125 size-[200px]",
                        {
                          "!rounded-full": im === noseStyle,
                        }
                      )}
                      src={`${protocol}//${hostname}:8001/get-image?im_name=${encodeURIComponent(
                        im
                      )}`}
                    />
                  ))}
              </div>
            </Modal>
            <Outlet />
          </div>
        </Content>
      </Layout>
    </Layout>
  )
}
