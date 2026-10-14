<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue'

const props = defineProps({
    game: {
        type: Object,
        required: true,
    },
    disabled: {
        type: Boolean,
        default: false,
    },
    actionMode: {
        type: String,
        default: 'move',
    },
})

const emit = defineEmits(['move', 'place-wall'])

const boardRef = ref(null)
const wallOrientation = ref('H')

const isDraggingWall = ref(false)
const dragTarget = ref(null)
const boardSizeVersion = ref(0)

const GRID_SIZE = 9
const WALL_GRID_SIZE = 8
const GRID_GAP = 5
const WALL_THICKNESS = 7

const cells = computed(() =>
    Array.from({ length: 81 }, (_, index) => ({
        x: index % GRID_SIZE,
        y: Math.floor(index / GRID_SIZE),
    })),
)

let pointerMoveHandler = null
let pointerUpHandler = null
let pointerCancelHandler = null
let resizeHandler = null

function pawnAt(x, y) {
    return props.game.pawns.findIndex(
        (pawn) => pawn.x === x && pawn.y === y,
    )
}

function isBlockedBetween(x1, y1, x2, y2) {
    const walls = props.game.walls ?? []

    if (x1 === x2) {
        const boundaryY = Math.min(y1, y2)

        return walls.some(
            (wall) =>
                wall.orientation === 'H' &&
                wall.y === boundaryY &&
                wall.x <= x1 &&
                x1 <= wall.x + 1,
        )
    }

    if (y1 === y2) {
        const boundaryX = Math.min(x1, x2)

        return walls.some(
            (wall) =>
                wall.orientation === 'V' &&
                wall.x === boundaryX &&
                wall.y <= y1 &&
                y1 <= wall.y + 1,
        )
    }

    return true
}

function isInsideBoard(x, y) {
    return x >= 0 && x < GRID_SIZE && y >= 0 && y < GRID_SIZE
}

function isLegalMove(x, y) {
    return getLegalPawnMoves().some(
        (move) => move.x === x && move.y === y,
    )
}

function getLegalPawnMoves() {
    const pawns = props.game.pawns ?? []
    const player = props.game.current_player

    const own = pawns[player]
    const opponent = pawns[1 - player]

    if (!own || !opponent || props.game.status !== 'in_progress') {
        return []
    }

    const moves = new Set()
    const directions = [
        [0, -1],
        [1, 0],
        [0, 1],
        [-1, 0],
    ]

    for (const [dx, dy] of directions) {
        const nx = own.x + dx
        const ny = own.y + dy

        if (!isInsideBoard(nx, ny)) continue

        if (isBlockedBetween(own.x, own.y, nx, ny)) continue

        // Petak di depan bukan posisi pion lawan.
        if (nx !== opponent.x || ny !== opponent.y) {
            moves.add(`${nx},${ny}`)
            continue
        }

        // Pion lawan bersebelahan. Coba melompat lurus.
        const jumpX = opponent.x + dx
        const jumpY = opponent.y + dy

        if (
            isInsideBoard(jumpX, jumpY) &&
            !isBlockedBetween(
                opponent.x,
                opponent.y,
                jumpX,
                jumpY,
            )
        ) {
            moves.add(`${jumpX},${jumpY}`)
            continue
        }

        // Lompatan lurus tidak tersedia.
        // Coba bergerak diagonal di samping pion lawan.
        for (const [sx, sy] of [
            [-dy, dx],
            [dy, -dx],
        ]) {
            const diagonalX = opponent.x + sx
            const diagonalY = opponent.y + sy

            if (!isInsideBoard(diagonalX, diagonalY)) continue

            if (
                isBlockedBetween(
                    opponent.x,
                    opponent.y,
                    diagonalX,
                    diagonalY,
                )
            ) {
                continue
            }

            moves.add(`${diagonalX},${diagonalY}`)
        }
    }

    return [...moves].map((position) => {
        const [x, y] = position.split(',').map(Number)
        return { x, y }
    })
}

function selectCell(x, y) {
    if (props.disabled) return
    if (props.actionMode !== 'move') return
    if (pawnAt(x, y) !== -1) return

    const legalMoves = getLegalPawnMoves()

    const isLegal = legalMoves.some(
        (move) => move.x === x && move.y === y,
    )

    if (!isLegal) return

    emit('move', [x, y])
}

/**
 * Mengukur grid berdasarkan ukuran aktual elemen papan.
 *
 * Tidak menggunakan persentase perkiraan.
 * Perhitungan mengikuti ukuran petak, gap, dan padding CSS.
 */
function getBoardMetrics() {
    // Membuat template menghitung ulang posisi saat papan berubah ukuran.
    void boardSizeVersion.value

    const element = boardRef.value

    if (!element) return null

    const rect = element.getBoundingClientRect()
    const css = window.getComputedStyle(element)

    const borderLeft = parseFloat(css.borderLeftWidth) || 0
    const borderRight = parseFloat(css.borderRightWidth) || 0
    const borderTop = parseFloat(css.borderTopWidth) || 0
    const borderBottom = parseFloat(css.borderBottomWidth) || 0

    const paddingLeft = parseFloat(css.paddingLeft) || 0
    const paddingRight = parseFloat(css.paddingRight) || 0
    const paddingTop = parseFloat(css.paddingTop) || 0
    const paddingBottom = parseFloat(css.paddingBottom) || 0

    const innerWidth =
        rect.width -
        borderLeft -
        borderRight -
        paddingLeft -
        paddingRight

    const innerHeight =
        rect.height -
        borderTop -
        borderBottom -
        paddingTop -
        paddingBottom

    const cellWidth =
        (innerWidth - GRID_GAP * (GRID_SIZE - 1)) / GRID_SIZE

    const cellHeight =
        (innerHeight - GRID_GAP * (GRID_SIZE - 1)) / GRID_SIZE

    return {
        rect,
        borderLeft,
        borderTop,
        paddingLeft,
        paddingTop,
        innerWidth,
        innerHeight,
        cellWidth,
        cellHeight,
        stepX: cellWidth + GRID_GAP,
        stepY: cellHeight + GRID_GAP,

        // Titik awal area grid dalam koordinat layar.
        gridLeft: rect.left + borderLeft + paddingLeft,
        gridTop: rect.top + borderTop + paddingTop,
    }
}

/**
 * Mengubah koordinat pointer menjadi posisi dinding.
 *
 * H:
 *   x = awal dinding secara horizontal
 *   y = batas horizontal di antara dua baris
 *
 * V:
 *   x = batas vertikal di antara dua kolom
 *   y = awal dinding secara vertikal
 */
function getWallTarget(clientX, clientY) {
    const metrics = getBoardMetrics()

    if (!metrics) return null

    const {
        gridLeft,
        gridTop,
        innerWidth,
        innerHeight,
        cellWidth,
        cellHeight,
        stepX,
        stepY,
    } = metrics

    const localX = clientX - gridLeft
    const localY = clientY - gridTop

    // Jangan melakukan snap ketika pointer berada di luar papan.
    if (
        localX < 0 ||
        localY < 0 ||
        localX > innerWidth ||
        localY > innerHeight
    ) {
        return null
    }

    let x
    let y

    if (wallOrientation.value === 'H') {
        // Awal dinding mengikuti kolom.
        x = Math.floor(localX / stepX)

        // Dinding horizontal mengikuti celah ant baris.
        y = Math.round(
            (localY - cellHeight - GRID_GAP / 2) / stepY,
        )
    } else {
        // Dinding vertikal mengikuti celah antarkolom.
        x = Math.round(
            (localX - cellWidth - GRID_GAP / 2) / stepX,
        )

        // Awal dinding mengikuti baris.
        y = Math.floor(localY / stepY)
    }

    // Papan 9x9 memiliki 8x8 posisi awal dinding.
    x = Math.max(0, Math.min(WALL_GRID_SIZE - 1, x))
    y = Math.max(0, Math.min(WALL_GRID_SIZE - 1, y))

    return {
        orientation: wallOrientation.value,
        x,
        y,
    }
}

/**
 * Satu fungsi posisi untuk dinding yang sudah terpasang
 * maupun pratinjau selama drag.
 */
function getWallStyle(wall) {
    const metrics = getBoardMetrics()

    if (!metrics) return {}

    const {
        paddingLeft,
        paddingTop,
        cellWidth,
        cellHeight,
        stepX,
        stepY,
    } = metrics

    if (wall.orientation === 'H') {
        return {
            left: `${paddingLeft + wall.x * stepX}px`,

            top: `${paddingTop +
                wall.y * stepY +
                cellHeight +
                GRID_GAP / 2 -
                WALL_THICKNESS / 2
                }px`,

            width: `${cellWidth * 2 + GRID_GAP}px`,
            height: `${WALL_THICKNESS}px`,
        }
    }

    return {
        left: `${paddingLeft +
            wall.x * stepX +
            cellWidth +
            GRID_GAP / 2 -
            WALL_THICKNESS / 2
            }px`,

        top: `${paddingTop + wall.y * stepY}px`,

        width: `${WALL_THICKNESS}px`,
        height: `${cellHeight * 2 + GRID_GAP}px`,
    }
}

function updateDrag(event) {
    if (!isDraggingWall.value) return

    dragTarget.value = getWallTarget(
        event.clientX,
        event.clientY,
    )
}

function removePointerListeners() {
    if (pointerMoveHandler) {
        window.removeEventListener(
            'pointermove',
            pointerMoveHandler,
        )
    }

    if (pointerUpHandler) {
        window.removeEventListener(
            'pointerup',
            pointerUpHandler,
        )
    }

    if (pointerCancelHandler) {
        window.removeEventListener(
            'pointercancel',
            pointerCancelHandler,
        )
    }

    pointerMoveHandler = null
    pointerUpHandler = null
    pointerCancelHandler = null
}

function cancelWallDrag() {
    removePointerListeners()

    isDraggingWall.value = false
    dragTarget.value = null
}

function finishWallDrag(event) {
    // Hitung posisi terakhir sebelum menyelesaikan drag.
    updateDrag(event)

    const target = dragTarget.value

    removePointerListeners()

    isDraggingWall.value = false
    dragTarget.value = null

    if (props.disabled || !target) return

    emit('place-wall', {
        orientation: target.orientation,
        x: target.x,
        y: target.y,
    })
}

function startWallDrag(event, orientation) {
    if (props.disabled || isDraggingWall.value) return

    if (orientation !== 'H' && orientation !== 'V') return

    wallOrientation.value = orientation
    isDraggingWall.value = true

    // Hanya gunakan tombol utama mouse.
    if (event.pointerType === 'mouse' && event.button !== 0) {
        return
    }

    event.preventDefault()

    // Bersihkan listener drag sebelumnya jika ada.
    removePointerListeners()

    isDraggingWall.value = true
    dragTarget.value = getWallTarget(
        event.clientX,
        event.clientY,
    )

    pointerMoveHandler = updateDrag
    pointerUpHandler = finishWallDrag
    pointerCancelHandler = cancelWallDrag

    window.addEventListener(
        'pointermove',
        pointerMoveHandler,
    )

    window.addEventListener(
        'pointerup',
        pointerUpHandler,
    )

    window.addEventListener(
        'pointercancel',
        pointerCancelHandler,
    )
}

const wallPreviewStyle = computed(() => {
    if (!isDraggingWall.value || !dragTarget.value) {
        return {}
    }

    return getWallStyle(dragTarget.value)
})

onMounted(() => {
    resizeHandler = () => {
        boardSizeVersion.value++
    }

    window.addEventListener('resize', resizeHandler)
})

onUnmounted(() => {
    removePointerListeners()

    if (resizeHandler) {
        window.removeEventListener('resize', resizeHandler)
    }
})
</script>

<template>
    <div class="board-wrapper">
        <div v-if="actionMode === 'wall'" class="wall-toolbar">
            <span>Orientasi dinding:</span>

            <button type="button" :class="{ active: wallOrientation === 'H' }" :disabled="disabled || isDraggingWall"
                @pointerdown="startWallDrag($event, 'H')">
                <span class="wall-drag-icon icon-horizontal"></span>
                Horizontal
            </button>

            <button type="button" :class="{ active: wallOrientation === 'V' }" :disabled="disabled || isDraggingWall"
                @pointerdown="startWallDrag($event, 'V')">
                <span class="wall-drag-icon icon-vertical"></span>
                Vertikal
            </button>

            <span class="wall-hint">
                Tahan tombol di atas, seret ke celah grid, lalu lepaskan.
            </span>
        </div>

        <div ref="boardRef" class="board" :class="{
            'wall-dragging': isDraggingWall,
        }">
            <button v-for="cell in cells" :key="`${cell.x}-${cell.y}`" type="button" class="cell" :class="{
                'pawn-human':
                    pawnAt(cell.x, cell.y) === game.human_player,

                'pawn-ai':
                    pawnAt(cell.x, cell.y) === game.ai_player,

                'valid-move':
                    !disabled &&
                    actionMode === 'move' &&
                    pawnAt(cell.x, cell.y) === -1 &&
                    isLegalMove(cell.x, cell.y),
            }" :disabled="disabled" :aria-label="`Petak ${cell.x}, ${cell.y}`" @click="selectCell(cell.x, cell.y)">
                <span v-if="pawnAt(cell.x, cell.y) === game.human_player" class="pawn human">
                    P
                </span>

                <span v-else-if="pawnAt(cell.x, cell.y) === game.ai_player" class="pawn ai">
                    AI
                </span>
            </button>

            <!-- Dinding yang sudah terpasang -->
            <div v-for="(wall, index) in game.walls" :key="`wall-${index}`" class="wall" :class="wall.orientation === 'H'
                ? 'wall-horizontal'
                : 'wall-vertical'
                " :style="getWallStyle(wall)"></div>

            <!-- Pratinjau dinding saat drag -->
            <div v-if="isDraggingWall && dragTarget" class="wall wall-preview" :class="dragTarget.orientation === 'H'
                ? 'wall-horizontal'
                : 'wall-vertical'
                " :style="wallPreviewStyle"></div>

            <div v-if="isDraggingWall && !dragTarget" class="drag-outside-hint">
                Arahkan dinding ke papan
            </div>
        </div>
    </div>
</template>

<style scoped>
.board-wrapper {
    width: 100%;
    max-width: 620px;
    margin: 0 auto;
    user-select: none;
}

.wall-toolbar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    margin-bottom: 14px;
    color: #cbd5e1;
    font-size: 0.85rem;
}

.wall-toolbar button {
    padding: 7px 12px;
    border: 1px solid #475569;
    border-radius: 7px;
    background: #1e293b;
    color: #e5e7eb;
    cursor: pointer;
}

.wall-toolbar button.active {
    border-color: #fbbf24;
    background: #78350f;
}

.wall-toolbar button:disabled {
    opacity: 0.55;
    cursor: not-allowed;
}

.wall-drag-handle {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    border-color: #d97706 !important;
    background: #92400e !important;
    cursor: grab !important;
    touch-action: none;
}

.wall-drag-handle:active {
    cursor: grabbing !important;
}

.wall-drag-icon {
    display: inline-block;
    flex: 0 0 auto;
    border-radius: 3px;
    background: #fbbf24;
}

.wall-drag-icon.icon-horizontal {
    width: 24px;
    height: 7px;
}

.wall-drag-icon.icon-vertical {
    width: 7px;
    height: 24px;
}

.wall-hint {
    flex-basis: 100%;
    color: #9ca3af;
}

.board {
    position: relative;
    display: grid;
    grid-template-columns: repeat(9, minmax(0, 1fr));
    gap: 5px;
    padding: 8px;
    border: 1px solid #475569;
    border-radius: 12px;
    background: #0f172a;
    aspect-ratio: 1;
    box-sizing: border-box;
    isolation: isolate;
}

.cell {
    display: grid;
    min-width: 0;
    aspect-ratio: 1;
    place-items: center;
    padding: 0;
    border: 1px solid #334155;
    border-radius: 5px;
    background: #1e293b;
    color: #f8fafc;
    cursor: pointer;
    box-sizing: border-box;
}

.cell:disabled {
    cursor: not-allowed;
}

.cell.valid-move {
    border-color: #34d399;
    background: #14532d;
}

.pawn {
    display: grid;
    width: 78%;
    height: 78%;
    place-items: center;
    border-radius: 50%;
    font-size: clamp(9px, 1.5vw, 15px);
    font-weight: 800;
    pointer-events: none;
}

.pawn.human {
    background: #f59e0b;
    color: #111827;
}

.pawn.ai {
    background: #60a5fa;
    color: #111827;
}

/* Semua dinding memakai koordinat absolut dari fungsi getWallStyle(). */
.wall {
    position: absolute;
    z-index: 5;
    border-radius: 3px;
    background: #fbbf24;
    pointer-events: none;
    box-sizing: border-box;
}

.wall-preview {
    z-index: 10;
    background: rgba(251, 191, 36, 0.8);
    outline: 2px solid rgba(255, 255, 255, 0.75);
    box-shadow: 0 0 12px rgba(251, 191, 36, 0.35);
}

.wall-dragging {
    cursor: grabbing;
}

.drag-outside-hint {
    position: absolute;
    z-index: 20;
    top: 50%;
    left: 50%;
    padding: 8px 12px;
    border: 1px solid #475569;
    border-radius: 8px;
    background: rgba(15, 23, 42, 0.95);
    color: #f8fafc;
    font-size: 0.85rem;
    pointer-events: none;
    transform: translate(-50%, -50%);
}
</style>