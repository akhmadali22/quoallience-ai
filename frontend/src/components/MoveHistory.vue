<script setup>
import { computed, ref } from 'vue'

const props = defineProps({
    history: {
        type: Array,
        default: () => [],
    },
})

const copied = ref(false)
const copyError = ref('')
let copiedTimeout = null

const formattedHistory = computed(() => {
    return props.history.map((item, index) => {
        const action = item.action ?? {}

        let description = 'Aksi tidak diketahui'

        if (action.type === 'move') {
            const destination = action.destination ?? []
            description = `Bergerak ke (${destination[0]}, ${destination[1]})`
        } else if (action.type === 'wall') {
            description = `Memasang dinding ${action.orientation ?? '?'} di (${action.x}, ${action.y})`
        }

        return {
            id: `${item.ply ?? index + 1}-${item.player ?? 'unknown'}`,
            ply: item.ply ?? index + 1,
            player: item.player,
            description,
            analysis: item.analysis,
        }
    })
})

const exportHistory = computed(() => {
    return {
        total_act: formattedHistory.value.length,
        act_list: formattedHistory.value.map((item) => {
            const playerLabel =
                item.player === 0 || item.player === 1
                    ? `Pemain ${item.player}`
                    : 'Pemain'

            let entry = `${item.ply}. ${playerLabel} ${item.description}`

            if (item.analysis) {
                const depth = item.analysis.depth ?? '-'
                const elapsed = item.analysis.elapsed_ms ?? 0

                entry += ` Depth ${depth} · ${elapsed} ms`
            }

            return entry
        }),
    }
})

async function copyHistoryToClipboard() {
    copyError.value = ''

    try {
        const json = JSON.stringify(exportHistory.value, null, 2)

        await navigator.clipboard.writeText(json)

        copied.value = true

        if (copiedTimeout) {
            clearTimeout(copiedTimeout)
        }

        copiedTimeout = setTimeout(() => {
            copied.value = false
            copiedTimeout = null
        }, 2000)
    } catch (error) {
        console.error('Gagal menyalin riwayat pertandingan:', error)

        copyError.value =
            'Gagal menyalin. Pastikan browser mengizinkan akses clipboard.'
    }
}
</script>

<template>
    <section class="history-panel">
        <header class="panel-header">
            <div>
                <h2>Riwayat langkah</h2>
                <p>{{ history.length }} aksi tercatat</p>
            </div>
            <button type="button" class="copy-button" :disabled="history.length === 0" @click="copyHistoryToClipboard">
                {{ copied ? 'Tersalin!' : 'Salin JSON' }}
            </button>
        </header>
        <p v-if="copyError" class="copy-error" role="alert">
            {{ copyError }}
        </p>
        <div v-if="formattedHistory.length === 0" class="empty-state">
            Belum ada langkah. Pertandingan baru dimulai.
        </div>

        <ol v-else class="history-list">
            <li v-for="item in formattedHistory" :key="item.id" class="history-item">
                <span class="move-number">{{ item.ply }}</span>

                <div class="move-details">
                    <strong>
                        {{
                            item.player === 0
                                ? 'Pemain 0'
                                : item.player === 1
                                    ? 'Pemain 1'
                                    : 'Pemain'
                        }}
                    </strong>

                    <span>{{ item.description }}</span>

                    <small v-if="item.analysis">
                        Depth {{ item.analysis.depth ?? '-' }}
                        · {{ item.analysis.elapsed_ms ?? 0 }} ms
                    </small>
                </div>
            </li>
        </ol>
    </section>
</template>

<style scoped>
.history-panel {
    padding: 20px;
    color: #e5e7eb;
}

.panel-header {
    margin-bottom: 16px;
}

.panel-header h2 {
    margin: 0;
    font-size: 1.05rem;
}

.panel-header p {
    margin: 5px 0 0;
    color: #9ca3af;
    font-size: 0.85rem;
}

.empty-state {
    padding: 24px 8px;
    color: #9ca3af;
    font-size: 0.9rem;
    text-align: center;
}

.history-list {
    display: grid;
    gap: 12px;
    max-height: 320px;
    margin: 0;
    padding: 0;
    overflow-y: auto;
    list-style: none;
}

.history-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding-bottom: 12px;
    border-bottom: 1px solid #374151;
}

.move-number {
    display: grid;
    width: 28px;
    height: 28px;
    flex-shrink: 0;
    place-items: center;
    border-radius: 8px;
    background: #1f2937;
    color: #d1d5db;
    font-size: 0.8rem;
}

.move-details {
    display: grid;
    gap: 4px;
    font-size: 0.85rem;
}

.move-details strong {
    color: #f9fafb;
}

.move-details>span {
    color: #d1d5db;
}

.move-details small {
    color: #9ca3af;
}

.panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 16px;
}

.copy-button {
    flex-shrink: 0;
    padding: 8px 12px;
    border: 1px solid #4b5563;
    border-radius: 8px;
    background: #1f2937;
    color: #e5e7eb;
    font-size: 0.8rem;
    font-weight: 600;
    cursor: pointer;
    transition:
        background 0.2s ease,
        border-color 0.2s ease;
}

.copy-button:hover:not(:disabled) {
    background: #374151;
    border-color: #6b7280;
}

.copy-button:disabled {
    opacity: 0.4;
    cursor: not-allowed;
}

.copy-error {
    margin: 0 0 12px;
    color: #fca5a5;
    font-size: 0.8rem;
}
</style>