"""
WSI Tile Server for MOSAIC

Provides DeepZoom-compatible tile serving for whole slide images using OpenSlide.
Supports dynamic tile generation with caching for frequently accessed tiles.
"""

import hashlib
import io
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    from openslide import OpenSlide
    from openslide.deepzoom import DeepZoomGenerator

    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False
    OpenSlide = None
    DeepZoomGenerator = None

from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WSITileServer:
    """
    Tile server for whole slide images with DeepZoom support.

    Provides efficient tile serving with caching for web-based WSI viewers.
    """

    # DeepZoom configuration
    TILE_SIZE = 256
    OVERLAP = 1
    FORMAT = "jpeg"
    QUALITY = 85

    def __init__(
        self,
        wsi_directory: str = "data/raw/svs",
        upload_directory: str = "data/uploads",
        cache_size: int = 1000,
        tile_size: int = 256,
        overlap: int = 1,
    ):
        """
        Initialize the tile server.

        Args:
            wsi_directory: Directory containing SVS/WSI files
            upload_directory: Directory containing uploaded WSI files
            cache_size: Number of tiles to cache in memory
            tile_size: Size of each tile in pixels
            overlap: Overlap between adjacent tiles
        """
        if not OPENSLIDE_AVAILABLE:
            raise ImportError(
                "OpenSlide is required for WSI tile serving. "
                "Install with: pip install openslide-python"
            )

        self.wsi_directory = Path(wsi_directory)
        self.upload_directory = Path(upload_directory)
        self.tile_size = tile_size
        self.overlap = overlap
        self.cache_size = cache_size

        # Cache for opened slides
        self._slide_cache: Dict[str, Tuple[OpenSlide, DeepZoomGenerator]] = {}

        # LRU cache for tiles
        self._get_tile_cached = lru_cache(maxsize=cache_size)(self._get_tile_impl)

        logger.info(
            f"WSI Tile Server initialized. Directories: {self.wsi_directory}, {self.upload_directory}"
        )

    def _find_slide_path(self, slide_id: str) -> Optional[Path]:
        """
        Find the full path to a slide file.

        Args:
            slide_id: Slide identifier (filename without extension or with)

        Returns:
            Path to slide file or None if not found
        """
        # Common WSI extensions
        extensions = [".svs", ".tif", ".tiff", ".ndpi", ".vms", ".vmu", ".scn", ".mrxs"]

        # Search in both wsi_directory and upload_directory
        search_dirs = [self.wsi_directory, self.upload_directory]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue

            # Check if slide_id already has extension
            slide_path = search_dir / slide_id
            if slide_path.exists():
                return slide_path

            # Try with common extensions
            for ext in extensions:
                candidate = search_dir / f"{slide_id}{ext}"
                if candidate.exists():
                    return candidate

            # Search recursively
            for ext in extensions:
                matches = list(search_dir.rglob(f"*{slide_id}*{ext}"))
                if matches:
                    return matches[0]

        return None

    def _get_slide(self, slide_id: str) -> Tuple[OpenSlide, DeepZoomGenerator]:
        """
        Get or create OpenSlide and DeepZoom objects for a slide.

        Args:
            slide_id: Slide identifier

        Returns:
            Tuple of (OpenSlide, DeepZoomGenerator)
        """
        if slide_id in self._slide_cache:
            return self._slide_cache[slide_id]

        slide_path = self._find_slide_path(slide_id)
        if slide_path is None:
            raise FileNotFoundError(f"Slide not found: {slide_id}")

        slide = OpenSlide(str(slide_path))
        dz = DeepZoomGenerator(
            slide, tile_size=self.tile_size, overlap=self.overlap, limit_bounds=True
        )

        # Manage cache size
        if len(self._slide_cache) >= 10:  # Keep max 10 slides open
            # Remove oldest entry
            oldest = next(iter(self._slide_cache))
            old_slide, _ = self._slide_cache.pop(oldest)
            old_slide.close()

        self._slide_cache[slide_id] = (slide, dz)
        return slide, dz

    def _get_tile_impl(
        self,
        cache_key: str,  # Used for caching: f"{slide_id}_{level}_{col}_{row}"
        slide_id: str,
        level: int,
        col: int,
        row: int,
        format: str = "jpeg",
    ) -> bytes:
        """
        Internal tile generation (cached).

        Args:
            cache_key: Unique key for caching
            slide_id: Slide identifier
            level: DeepZoom level
            col: Tile column
            row: Tile row
            format: Output format ('jpeg' or 'png')

        Returns:
            Tile image as bytes
        """
        _, dz = self._get_slide(slide_id)

        # Validate coordinates
        if level >= dz.level_count:
            raise ValueError(f"Invalid level {level}. Max: {dz.level_count - 1}")

        level_tiles = dz.level_tiles[level]
        if col >= level_tiles[0] or row >= level_tiles[1]:
            raise ValueError(
                f"Invalid tile coordinates ({col}, {row}) for level {level}. "
                f"Max: ({level_tiles[0] - 1}, {level_tiles[1] - 1})"
            )

        # Generate tile
        tile = dz.get_tile(level, (col, row))

        # Convert to bytes
        buffer = io.BytesIO()
        if format.lower() in ["jpg", "jpeg"]:
            tile.save(buffer, format="JPEG", quality=self.QUALITY)
        else:
            tile.save(buffer, format="PNG")

        return buffer.getvalue()

    def get_tile(
        self, slide_id: str, level: int, col: int, row: int, format: str = "jpeg"
    ) -> bytes:
        """
        Get a tile from a WSI.

        Args:
            slide_id: Slide identifier
            level: DeepZoom level (0 = most zoomed out)
            col: Tile column index
            row: Tile row index
            format: Output format ('jpeg' or 'png')

        Returns:
            Tile image as bytes
        """
        cache_key = f"{slide_id}_{level}_{col}_{row}_{format}"
        return self._get_tile_cached(cache_key, slide_id, level, col, row, format)

    def get_dzi_metadata(self, slide_id: str) -> dict:
        """
        Get DeepZoom Image (DZI) metadata for a slide.

        Args:
            slide_id: Slide identifier

        Returns:
            Dictionary with DZI metadata
        """
        slide, dz = self._get_slide(slide_id)

        return {
            "slide_id": slide_id,
            "dimensions": slide.dimensions,
            "level_count": dz.level_count,
            "level_dimensions": dz.level_dimensions,
            "level_tiles": dz.level_tiles,
            "tile_size": self.tile_size,
            "overlap": self.overlap,
            "format": self.FORMAT,
            "properties": dict(slide.properties),
        }

    def get_dzi_xml(self, slide_id: str) -> str:
        """
        Get DeepZoom Image XML descriptor for OpenSeadragon compatibility.

        Args:
            slide_id: Slide identifier

        Returns:
            DZI XML string
        """
        _, dz = self._get_slide(slide_id)

        width, height = dz.level_dimensions[-1]  # Highest resolution

        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"
       Format="{self.FORMAT}"
       Overlap="{self.overlap}"
       TileSize="{self.tile_size}">
    <Size Width="{width}" Height="{height}"/>
</Image>"""

        return xml

    def get_thumbnail(
        self, slide_id: str, max_size: Tuple[int, int] = (512, 512)
    ) -> bytes:
        """
        Get a thumbnail of the slide.

        Args:
            slide_id: Slide identifier
            max_size: Maximum thumbnail dimensions (width, height)

        Returns:
            Thumbnail image as JPEG bytes
        """
        slide, _ = self._get_slide(slide_id)

        # Get thumbnail from OpenSlide
        thumb = slide.get_thumbnail(max_size)

        buffer = io.BytesIO()
        thumb.save(buffer, format="JPEG", quality=90)
        return buffer.getvalue()

    def list_slides(self) -> list:
        """
        List all available slides.

        Returns:
            List of slide identifiers
        """
        extensions = ["*.svs", "*.tif", "*.tiff", "*.ndpi", "*.vms", "*.scn", "*.mrxs"]
        slides = []

        # Search in both directories
        search_dirs = [self.wsi_directory, self.upload_directory]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for ext in extensions:
                for path in search_dir.rglob(ext):
                    slides.append(path.stem)

        return sorted(set(slides))

    def close(self):
        """Close all open slides."""
        if not hasattr(self, "_slide_cache"):
            return
        for slide_id, (slide, _) in self._slide_cache.items():
            try:
                slide.close()
            except:
                pass
        self._slide_cache.clear()
        logger.info("All slides closed")

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, "_slide_cache"):
            self.close()


# Singleton instance for FastAPI
_tile_server_instance: Optional[WSITileServer] = None


def get_tile_server(
    wsi_directory: str = "data/raw/svs", upload_directory: str = "data/uploads"
) -> WSITileServer:
    """
    Get or create the global tile server instance.

    Args:
        wsi_directory: Directory containing WSI files
        upload_directory: Directory containing uploaded WSI files

    Returns:
        WSITileServer instance
    """
    global _tile_server_instance

    if _tile_server_instance is None:
        try:
            _tile_server_instance = WSITileServer(
                wsi_directory=wsi_directory, upload_directory=upload_directory
            )
        except ImportError as e:
            logger.warning(f"WSI Tile Server unavailable: {e}")
            return None

    return _tile_server_instance
