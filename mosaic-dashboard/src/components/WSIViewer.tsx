import { useEffect, useRef, useState } from 'react'
import OpenSeadragon from 'openseadragon'
import {
  Box,
  Paper,
  Typography,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  CircularProgress,
  Alert,
  Slider,
  FormControlLabel,
  Switch,
} from '@mui/material'
import { getSlideList, getDziUrl } from '../api'

interface WSIViewerProps {
  slideId?: string | null
  attentionOverlay?: number[][] | null
}

export default function WSIViewer({ slideId, attentionOverlay }: WSIViewerProps) {
  const viewerRef = useRef<HTMLDivElement>(null)
  const osdRef = useRef<OpenSeadragon.Viewer | null>(null)
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null)
  
  const [availableSlides, setAvailableSlides] = useState<string[]>([])
  const [selectedSlide, setSelectedSlide] = useState<string>(slideId || '')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showOverlay, setShowOverlay] = useState(true)
  const [overlayOpacity, setOverlayOpacity] = useState(0.5)

  // Fetch available slides on mount
  useEffect(() => {
    async function fetchSlides() {
      try {
        const slides = await getSlideList()
        setAvailableSlides(slides)
        if (slides.length > 0 && !selectedSlide) {
          setSelectedSlide(slides[0])
        }
      } catch (err) {
        console.error('Failed to fetch slides:', err)
        setError('Failed to load available slides. WSI server may not be available.')
      }
    }
    fetchSlides()
  }, [])

  // Update selected slide when prop changes
  useEffect(() => {
    if (slideId && slideId !== selectedSlide) {
      setSelectedSlide(slideId)
    }
  }, [slideId])

  // Initialize/update OpenSeadragon viewer
  useEffect(() => {
    if (!viewerRef.current || !selectedSlide) return

    setLoading(true)
    setError(null)

    // Destroy existing viewer
    if (osdRef.current) {
      osdRef.current.destroy()
    }

    // Create new viewer
    const viewer = OpenSeadragon({
      element: viewerRef.current,
      prefixUrl: 'https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/images/',
      tileSources: getDziUrl(selectedSlide),
      showNavigator: true,
      navigatorPosition: 'BOTTOM_RIGHT',
      navigatorSizeRatio: 0.15,
      showRotationControl: true,
      gestureSettingsMouse: {
        clickToZoom: true,
        dblClickToZoom: true,
      },
      animationTime: 0.5,
      blendTime: 0.1,
      constrainDuringPan: true,
      maxZoomPixelRatio: 2,
      minZoomLevel: 0.1,
      visibilityRatio: 0.5,
      zoomPerScroll: 1.2,
    })

    viewer.addHandler('open', () => {
      setLoading(false)
      // Add overlay canvas after image loads
      addOverlayCanvas(viewer)
    })

    viewer.addHandler('open-failed', (event) => {
      setLoading(false)
      setError(`Failed to load slide: ${event.message}`)
    })

    osdRef.current = viewer

    return () => {
      viewer.destroy()
    }
  }, [selectedSlide])

  // Add overlay canvas for attention heatmap
  const addOverlayCanvas = (viewer: OpenSeadragon.Viewer) => {
    if (!attentionOverlay || !showOverlay) return

    // Create canvas overlay
    const canvas = document.createElement('canvas')
    canvas.style.position = 'absolute'
    canvas.style.top = '0'
    canvas.style.left = '0'
    canvas.style.pointerEvents = 'none'
    canvas.style.opacity = String(overlayOpacity)
    
    overlayCanvasRef.current = canvas

    // Add to viewer
    viewer.canvas.appendChild(canvas)

    // Draw attention heatmap
    drawAttentionHeatmap(viewer, canvas)

    // Update on viewport change
    viewer.addHandler('viewport-change', () => {
      drawAttentionHeatmap(viewer, canvas)
    })
  }

  // Draw attention heatmap on canvas
  const drawAttentionHeatmap = (
    viewer: OpenSeadragon.Viewer,
    canvas: HTMLCanvasElement
  ) => {
    if (!attentionOverlay) return

    const viewportRect = viewer.viewport.getBounds()
    const containerSize = viewer.viewport.getContainerSize()

    canvas.width = containerSize.x
    canvas.height = containerSize.y

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Map attention values to colors (red = high attention)
    // This is a simplified version - real implementation would
    // need proper coordinate mapping based on patch locations
    const gridSize = Math.ceil(Math.sqrt(attentionOverlay.length))
    const cellWidth = canvas.width / gridSize
    const cellHeight = canvas.height / gridSize

    attentionOverlay.forEach((row, i) => {
      if (Array.isArray(row)) {
        row.forEach((value, j) => {
          const normalizedValue = Math.min(1, Math.max(0, value))
          const hue = (1 - normalizedValue) * 240 // Blue to Red
          ctx.fillStyle = `hsla(${hue}, 100%, 50%, ${normalizedValue * 0.7})`
          ctx.fillRect(j * cellWidth, i * cellHeight, cellWidth, cellHeight)
        })
      }
    })
  }

  // Update overlay opacity
  useEffect(() => {
    if (overlayCanvasRef.current) {
      overlayCanvasRef.current.style.opacity = String(overlayOpacity)
    }
  }, [overlayOpacity])

  // Toggle overlay visibility
  useEffect(() => {
    if (overlayCanvasRef.current) {
      overlayCanvasRef.current.style.display = showOverlay ? 'block' : 'none'
    }
  }, [showOverlay])

  return (
    <Box sx={{ height: '600px', display: 'flex', flexDirection: 'column' }}>
      {/* Controls */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 3, flexWrap: 'wrap' }}>
          <FormControl sx={{ minWidth: 200 }}>
            <InputLabel>Select Slide</InputLabel>
            <Select
              value={selectedSlide}
              label="Select Slide"
              onChange={(e) => setSelectedSlide(e.target.value)}
              disabled={availableSlides.length === 0}
            >
              {availableSlides.map((slide) => (
                <MenuItem key={slide} value={slide}>
                  {slide}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {attentionOverlay && (
            <>
              <FormControlLabel
                control={
                  <Switch
                    checked={showOverlay}
                    onChange={(e) => setShowOverlay(e.target.checked)}
                  />
                }
                label="Show Attention Overlay"
              />

              <Box sx={{ width: 200 }}>
                <Typography variant="caption">Overlay Opacity</Typography>
                <Slider
                  value={overlayOpacity}
                  onChange={(_, value) => setOverlayOpacity(value as number)}
                  min={0}
                  max={1}
                  step={0.1}
                  disabled={!showOverlay}
                />
              </Box>
            </>
          )}
        </Box>
      </Paper>

      {/* Error Alert */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {/* Viewer Container */}
      <Box
        sx={{
          flexGrow: 1,
          position: 'relative',
          backgroundColor: '#1a1a1a',
          borderRadius: 1,
          overflow: 'hidden',
        }}
      >
        {loading && (
          <Box
            sx={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              zIndex: 10,
            }}
          >
            <CircularProgress />
          </Box>
        )}

        {!selectedSlide && !loading && (
          <Box
            sx={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              textAlign: 'center',
            }}
          >
            <Typography variant="h6" color="grey.500">
              No slide selected
            </Typography>
            <Typography variant="body2" color="grey.600">
              Select a slide from the dropdown above or upload one in the Patient Data tab
            </Typography>
          </Box>
        )}

        <div
          ref={viewerRef}
          style={{ width: '100%', height: '100%' }}
        />
      </Box>

      {/* Legend */}
      {attentionOverlay && showOverlay && (
        <Paper sx={{ p: 1, mt: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Typography variant="caption">Attention:</Typography>
            <Box
              sx={{
                width: 150,
                height: 15,
                background: 'linear-gradient(to right, blue, cyan, green, yellow, red)',
                borderRadius: 1,
              }}
            />
            <Typography variant="caption">Low</Typography>
            <Typography variant="caption">→</Typography>
            <Typography variant="caption">High</Typography>
          </Box>
        </Paper>
      )}
    </Box>
  )
}
