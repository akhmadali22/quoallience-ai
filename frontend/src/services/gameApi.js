import axios from "axios";

const api = axios.create({
  baseURL: "/api",
  headers: {
    "Content-Type": "application/json",
  },
});

export async function createGame({ humanPlayer = 0, firstPlayer = "human" } = {}) {
  const { data } = await api.post("/games", {
    human_player: humanPlayer,
    first_player: firstPlayer,
  });

  return data;
}

export async function getGame(gameId) {
  const { data } = await api.get(`/games/${gameId}`);
  return data;
}

export async function makeMove(gameId, destination) {
  const { data } = await api.post(`/games/${gameId}/move`, {
    destination,
  });

  return data;
}

export async function placeWall(gameId, orientation, x, y) {
  const { data } = await api.post(`/games/${gameId}/wall`, {
    orientation,
    x,
    y,
  });

  return data;
}

export async function makeAIMove(gameId) {
  const { data } = await api.post(`/games/${gameId}/ai-move`);
  return data;
}

export async function getMoveHistory(gameId) {
  const { data } = await api.get(`/games/${gameId}/history`);
  return data;
}

export async function deleteGame(gameId) {
  const { data } = await api.delete(`/games/${gameId}`);
  return data;
}
