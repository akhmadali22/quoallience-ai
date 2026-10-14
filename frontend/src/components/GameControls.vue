<script setup>
import { ref } from 'vue'

defineProps({
    loading: {
        type: Boolean,
        default: false,
    },
    actionMode: {
        type: String,
        default: 'move',
    },
})

const emit = defineEmits(['new-game', 'update:action-mode'])

const humanPlayer = ref(0)
const aiTimeLimit = ref(5)
const aiMaxDepth = ref(4)

function startNewGame() {
    emit('new-game', {
        humanPlayer: humanPlayer.value,
        aiTimeLimit: aiTimeLimit.value,
        aiMaxDepth: aiMaxDepth.value,
    })
}
</script>

<template>
    <section class="controls-panel">
        <h2>Kontrol permainan</h2>

        <label>
            Posisi pemain
            <select v-model.number="humanPlayer" :disabled="loading">
                <option :value="0">Pemain 0 · Bawah</option>
                <option :value="1">Pemain 1 · Atas</option>
            </select>
        </label>

        <label>
            Batas waktu AI (detik)
            <input v-model.number="aiTimeLimit" type="number" min="1" max="60" :disabled="loading" />
        </label>

        <label>
            Kedalaman maksimum AI
            <input v-model.number="aiMaxDepth" type="number" min="1" max="20" :disabled="loading" />
        </label>

        <div class="action-selector">
            <span>Mode aksi</span>

            <div class="action-buttons">
                <button :class="{ active: actionMode === 'move' }" :disabled="loading"
                    @click="emit('update:action-mode', 'move')">
                    Gerakkan pion
                </button>

                <button :class="{ active: actionMode === 'wall' }" :disabled="loading"
                    @click="emit('update:action-mode', 'wall')">
                    Pasang dinding
                </button>
            </div>

            <small>Mode dinding belum aktif pada tahap ini.</small>
        </div>

        <button class="new-game-button" :disabled="loading" @click="startNewGame">
            {{ loading ? 'Memproses...' : 'Permainan baru' }}
        </button>
    </section>
</template>

<style scoped>
.controls-panel {
    display: grid;
    gap: 16px;
    padding: 20px;
    color: #e5e7eb;
}

h2 {
    margin: 0;
    font-size: 1.05rem;
}

label {
    display: grid;
    gap: 7px;
    color: #cbd5e1;
    font-size: 0.88rem;
}

input,
select {
    width: 100%;
    padding: 10px;
    border: 1px solid #475569;
    border-radius: 8px;
    background: #1e293b;
    color: #f8fafc;
}

.action-selector {
    display: grid;
    gap: 8px;
    font-size: 0.88rem;
}

.action-buttons {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
}

.action-buttons button,
.new-game-button {
    padding: 10px;
    border: 1px solid #475569;
    border-radius: 8px;
    background: #1e293b;
    color: #e5e7eb;
    cursor: pointer;
}

.action-buttons button.active {
    border-color: #34d399;
    background: #064e3b;
}

.new-game-button {
    border-color: #047857;
    background: #047857;
    font-weight: 700;
}

button:disabled {
    cursor: not-allowed;
    opacity: 0.55;
}

small {
    color: #9ca3af;
}
</style>