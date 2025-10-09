import React, { useEffect, useState } from "react"
import { Popover, Progress, Typography, Space } from "antd"
import { CloudServerOutlined } from "@ant-design/icons"
import axios from "axios"

interface InversionStatus {
  total_steps: number
  current_step: number
}

const { Text } = Typography

const ServerStatus: React.FC = () => {
  const [data, setData] = useState<Record<string, InversionStatus>>({})
  const [loading, setLoading] = useState(false)

  const fetchData = async () => {
    try {
      setLoading(true)
      const res = await axios.get("processes")
      setData(res.data)
    } catch (err) {
      console.error("Failed to fetch process data", err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 3000)
    return () => clearInterval(interval)
  }, [])

  const renderContent = () => {
    const keys = Object.keys(data)
    if (keys.length === 0)
      return <Text type="secondary">No active processes</Text>

    return (
      <Space direction="vertical" style={{ width: 220 }}>
        {keys.map((key) => {
          const info = data[key]
          const { total_steps, current_step } = info || {}

          let percent = 0
          let status: "active" | "success" | "exception" = "active"

          if (current_step === -1) {
            percent = 100
            status = "exception"
          } else if (total_steps > 0) {
            percent = Math.min(
              100,
              Math.round((current_step / total_steps) * 100)
            )
            if (percent >= 100) status = "success"
          }

          return (
            <div key={key}>
              <Text>{key.replace("_inversion", "")}</Text>
              <Progress
                percent={percent}
                size="small"
                status={status}
                strokeColor={status === "exception" ? "red" : undefined}
              />
            </div>
          )
        })}
      </Space>
    )
  }

  return (
    <Popover content={renderContent()} trigger="hover" placement="bottomLeft">
      <CloudServerOutlined
        style={{
          fontSize: 24,
          color: loading ? "#1890ff" : "#52c41a",
          cursor: "pointer",
        }}
      />
    </Popover>
  )
}

export default ServerStatus
