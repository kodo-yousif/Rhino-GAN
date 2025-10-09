import React, { useRef } from "react"
import { Button, message } from "antd"
import { UploadOutlined } from "@ant-design/icons"
import axios from "axios"

const UploadImage: React.FC = () => {
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  const handleButtonClick = () => {
    fileInputRef.current?.click()
  }

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    const formData = new FormData()
    formData.append("image", file)

    try {
      await axios.post("upload-image", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      })
      message.success("Image uploaded successfully!")
    } catch (error) {
      console.error(error)
      message.error("Upload failed!")
    } finally {
      // reset the input so user can reselect same file later
      if (fileInputRef.current) {
        fileInputRef.current.value = ""
      }
    }
  }

  return (
    <>
      <input
        type="file"
        ref={fileInputRef}
        accept="image/*"
        onChange={handleFileChange}
        style={{ display: "none" }}
      />
      <Button
        icon={<UploadOutlined />}
        type="primary"
        onClick={handleButtonClick}
      >
        Upload Image
      </Button>
    </>
  )
}

export default UploadImage
