<script setup>
import { computed } from 'vue'

const props = defineProps({
    analysis: {
        type: Object,
        default: null,
    },
})

const statistics = computed(() => props.analysis?.statistics ?? {})

const metrics = computed(() => [
    {
        label: 'Kedalaman',
        value: props.analysis?.depth ?? '-',
    },
    {
        label: 'Waktu pencarian',
        value:
            props.analysis?.elapsed_ms !== undefined
                ? `${props.analysis.elapsed_ms} ms`
                : '-',
    },
    {
        label: 'Nodes',
        value: props.analysis?.nodes ?? '-',
    },
    {
        label: 'Alpha-beta cutoffs',
        value: statistics.value.alpha_beta_cutoffs ?? '-',
    },
    {
        label: 'Transposition table',
        value: statistics.value.transposition_table_entries ?? '-',
    },
    {
        label: 'Distance cache hits',
        value: statistics.value.distance_cache_hits ?? '-',
    },
    {
        label: 'Evaluation cache hits',
        value: statistics.value.evaluation_cache_hits ?? '-',
    },
])
</script>

<template>
    <section class="analysis-panel">
        <h2>Analisis AI</h2>

        <div v-if="!analysis" class="empty-state">
            Statistik pencarian akan muncul setelah AI bergerak.
        </div>

        <template v-else>
            <div class="score-card">
                <span>Skor evaluasi</span>
                <strong>{{ analysis.score }}</strong>
            </div>

            <div class="metrics-grid">
                <article v-for="metric in metrics" :key="metric.label" class="metric">
                    <span>{{ metric.label }}</span>
                    <strong>{{ metric.value }}</strong>
                </article>
            </div>

            <p v-if="analysis.timed_out" class="timeout-warning">
                Batas waktu tercapai sebelum pencarian selesai.
            </p>

            <div v-if="analysis.action" class="chosen-action">
                <span>Langkah yang dipilih</span>

                <strong v-if="analysis.action.type === 'move'">
                    Bergerak ke
                    {{ analysis.action.destination.join(', ') }}
                </strong>

                <strong v-else-if="analysis.action.type === 'wall'">
                    Dinding {{ analysis.action.orientation }} di
                    {{ analysis.action.x }}, {{ analysis.action.y }}
                </strong>
            </div>
        </template>
    </section>
</template>

<style scoped>
.analysis-panel {
    padding: 20px;
    color: #e5e7eb;
}

h2 {
    margin: 0 0 16px;
    font-size: 1.05rem;
}

.empty-state {
    color: #9ca3af;
    font-size: 0.9rem;
    line-height: 1.6;
}

.score-card {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
    padding: 14px;
    border: 1px solid #374151;
    border-radius: 10px;
    background: #1f2937;
}

.score-card span,
.metric span,
.chosen-action span {
    color: #9ca3af;
    font-size: 0.8rem;
}

.score-card strong {
    font-size: 1.4rem;
}

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
}

.metric {
    display: grid;
    gap: 7px;
    padding: 12px;
    border: 1px solid #374151;
    border-radius: 8px;
}

.metric strong {
    overflow-wrap: anywhere;
    font-size: 0.95rem;
}

.chosen-action {
    display: grid;
    gap: 7px;
    margin-top: 16px;
}

.timeout-warning {
    color: #fbbf24;
    font-size: 0.85rem;
}
</style>