<script setup>
import { computed } from 'vue'

const props = defineProps({
    game: {
        type: Object,
        required: true,
    },
})

const currentPlayerLabel = computed(() => {
    if (props.game.status !== 'in_progress') {
        return 'Permainan selesai'
    }

    return props.game.current_player === props.game.human_player
        ? 'Giliran kamu'
        : 'Giliran AI'
})

const winnerLabel = computed(() => {
    if (props.game.winner === null || props.game.winner === undefined) {
        return null
    }

    return props.game.winner === props.game.human_player
        ? 'Kamu menang!'
        : 'AI menang.'
})
</script>

<template>
    <section class="status-grid">
        <article class="status-card">
            <span>Status</span>
            <strong>{{ game.status }}</strong>
        </article>

        <article class="status-card">
            <span>Giliran</span>
            <strong>{{ currentPlayerLabel }}</strong>
        </article>

        <article class="status-card">
            <span>Dinding kamu</span>
            <strong>{{ game.walls_remaining[game.human_player] }}</strong>
        </article>

        <article class="status-card">
            <span>Dinding AI</span>
            <strong>{{ game.walls_remaining[game.ai_player] }}</strong>
        </article>

        <article v-if="winnerLabel" class="status-card winner">
            <strong>{{ winnerLabel }}</strong>
        </article>
    </section>
</template>

<style scoped>
.status-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
}

.status-card {
    display: grid;
    gap: 7px;
    padding: 16px;
    border: 1px solid #374151;
    border-radius: 12px;
    background: #111827;
    color: #e5e7eb;
}

.status-card span {
    color: #9ca3af;
    font-size: 0.8rem;
}

.status-card strong {
    font-size: 1rem;
}

.winner {
    border-color: #059669;
}
</style>