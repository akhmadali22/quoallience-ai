<script setup>
import { computed, onMounted, ref } from 'vue'
import {
    createGame,
    makeMove,
    makeAIMove,
    getMoveHistory,
    placeWall,
} from '../services/gameApi'

import QuoridorBoard from '../components/QuoridorBoard.vue'
import GameControls from '../components/GameControls.vue'
import GameStatus from '../components/GameStatus.vue'
import AIAnalysis from '../components/AIAnalysis.vue'
import MoveHistory from '../components/MoveHistory.vue'

const game = ref(null)
const aiAnalysis = ref(null)
const history = ref([])
const loading = ref(false)
const error = ref('')
const selectedAction = ref('move')

const isHumanTurn = computed(
    () =>
        game.value &&
        game.value.status === 'in_progress' &&
        game.value.current_player === game.value.human_player,
)

async function handlePlaceWall({ orientation, x, y }) {
    if (!isHumanTurn.value || loading.value) return

    loading.value = true
    error.value = ''
    aiAnalysis.value = null

    try {
        game.value = await placeWall(
            game.value.game_id,
            orientation,
            x,
            y,
        )

        if (game.value.status === 'in_progress') {
            await runAIMove()
        }

        await refreshHistory()
    } catch (err) {
        error.value =
            err.response?.data?.detail ??
            err.message ??
            'Dinding tidak dapat dipasang.'
    } finally {
        loading.value = false
    }
}

async function refreshHistory() {
    if (!game.value) return

    try {
        const result = await getMoveHistory(game.value.game_id)
        history.value = Array.isArray(result) ? result : result.history ?? []
    } catch (err) {
        console.error('Gagal mengambil riwayat:', err)
    }
}

async function startGame(options = {}) {
    loading.value = true
    error.value = ''
    aiAnalysis.value = null
    history.value = []

    try {
        game.value = await createGame(options)

        // Jika AI menjadi pemain pertama, jalankan langkah AI.
        if (game.value.current_player === game.value.ai_player) {
            await runAIMove()
        }
    } catch (err) {
        error.value =
            err.response?.data?.detail ??
            err.message ??
            'Gagal membuat permainan.'
    } finally {
        loading.value = false
    }
}

async function handleMove(destination) {
    if (!isHumanTurn.value || loading.value) return

    loading.value = true
    error.value = ''
    aiAnalysis.value = null

    try {
        game.value = await makeMove(game.value.game_id, destination)

        if (game.value.status === 'in_progress') {
            await runAIMove()
        }

        await refreshHistory()
    } catch (err) {
        error.value =
            err.response?.data?.detail ??
            err.message ??
            'Langkah tidak dapat dilakukan.'
    } finally {
        loading.value = false
    }
}

async function runAIMove() {
    if (!game.value || game.value.status !== 'in_progress') return

    const result = await makeAIMove(game.value.game_id)

    aiAnalysis.value = result
    game.value = result.game

    await refreshHistory()
}

onMounted(() => {
    startGame()
})
</script>

<template>
    <main class="game-page">
        <header class="page-header">
            <div>
                <h1>Quoridor AI</h1>
                <p>Strategi, navigasi, dan perebutan jalur menuju kemenangan.</p>
            </div>

            <span class="connection-status">
                {{ loading ? 'Memproses...' : 'API terhubung' }}
            </span>
        </header>

        <div v-if="error" class="error-message" role="alert">
            {{ error }}
        </div>

        <div v-if="loading && !game" class="loading-state">
            Membuat permainan...
        </div>

        <template v-else-if="game">
            <GameStatus :game="game" />

            <section class="game-layout">
                <div class="board-panel">
                    <div class="panel-heading">
                        <div>
                            <h2>Papan permainan</h2>
                            <p>
                                {{
                                    isHumanTurn
                                        ? 'Giliranmu. Pilih petak tujuan.'
                                        : 'AI sedang menentukan langkah.'
                                }}
                            </p>
                        </div>
                    </div>

                    <QuoridorBoard :game="game" :disabled="!isHumanTurn || loading" :action-mode="selectedAction"
                        @move="handleMove" @place-wall="handlePlaceWall" />

                    <div class="board-footer">
                        <span>Game ID: {{ game.game_id }}</span>
                        <span>Langkah: {{ game.ply }}</span>
                    </div>
                </div>

                <aside class="side-panel">
                    <GameControls :loading="loading" :action-mode="selectedAction" @new-game="startGame"
                        @update:action-mode="selectedAction = $event" />

                    <AIAnalysis :analysis="aiAnalysis" />

                    <MoveHistory :history="history" />
                </aside>
            </section>
        </template>
    </main>
</template>

<style scoped>
.game-page {
    max-width: 1440px;
    margin: 0 auto;
    padding: 28px;
    color: #e5e7eb;
}

.page-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
    margin-bottom: 24px;
}

.page-header h1 {
    margin: 0;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.04em;
}

.page-header p,
.panel-heading p {
    margin: 6px 0 0;
    color: #9ca3af;
}

.connection-status {
    padding: 7px 12px;
    border: 1px solid #374151;
    border-radius: 999px;
    color: #a7f3d0;
    font-size: 0.85rem;
}

.game-layout {
    display: grid;
    grid-template-columns: minmax(0, 1.3fr) minmax(300px, 0.7fr);
    gap: 24px;
    align-items: start;
}

.board-panel,
.side-panel>* {
    min-width: 0;
    border: 1px solid #374151;
    border-radius: 16px;
    background: #111827;
}

.board-panel {
    padding: 22px;
}

.panel-heading {
    margin-bottom: 20px;
}

.panel-heading h2 {
    margin: 0;
    font-size: 1.15rem;
}

.side-panel {
    display: grid;
    gap: 16px;
}

.board-footer {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    gap: 8px;
    margin-top: 16px;
    color: #9ca3af;
    font-size: 0.78rem;
}

.error-message {
    margin-bottom: 20px;
    padding: 12px 16px;
    border: 1px solid #991b1b;
    border-radius: 10px;
    background: #450a0a;
    color: #fecaca;
}

.loading-state {
    padding: 40px;
    text-align: center;
    color: #9ca3af;
}

@media (max-width: 900px) {
    .game-layout {
        grid-template-columns: 1fr;
    }

    .page-header {
        align-items: flex-start;
        flex-direction: column;
    }
}

@media (max-width: 600px) {
    .game-page {
        padding: 14px;
    }

    .board-panel {
        padding: 12px;
    }
}
</style>