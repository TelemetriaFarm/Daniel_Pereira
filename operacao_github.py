import pandas as pd
import numpy as np
import json
import requests
import time
from datetime import datetime, timedelta
from pytz import timezone
import math
from collections import Counter
import locale
import os
import sys

# --- CONFIGURAÇÕES DO USUÁRIO (PREENCHA AQUI) ---
# ID do Produtor (Grower ID)
GROWER_ID_FIXO = 1139788
CLIENTE_NOME = "Daniel Franco Pereira"

# IDs dos Implementos/Máquinas (Lista de inteiros)
# ATENÇÃO: Se IMPLEMENT_IDS_FIXO estiver vazio [], o script tentará buscar automaticamente
# todas as máquinas que contenham "Sprayer" ou "Pulverizador" no tipo.
IMPLEMENT_IDS_FIXO = [] 

# Tipos de máquina a serem buscados automaticamente (se IMPLEMENT_IDS_FIXO estiver vazio)
TIPO_MAQUINA_ALVO = ["Sprayer", "Pulverizador"] 

# Dados das Estações Climáticas (Hardcoded)
ESTACOES_DO_CLIENTE = [
    {'name': 'Fazenda Guará', 'id_estacao': '52944', 'latitude': -21.6533, 'longitude': -55.4610}
]
# ------------------------------------------------

# --- Importação da Autenticação ---
try:
    from farm_auth import get_authenticated_session
except ImportError:
    print("❌ ERRO CRÍTICO: Não foi possível encontrar o arquivo 'farm_auth.py'.")
    sys.exit(1)

# --- Configuração de Locale ---
try:
    locale.setlocale(locale.LC_TIME, 'pt_BR.UTF-8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_TIME, 'Portuguese_Brazil.1252')
    except locale.Error:
        print("AVISO: Locale português não disponível. Usando padrão.")

# --- Bibliotecas Geográficas ---
try:
    from shapely.geometry import LineString
    from shapely.ops import unary_union
    from pyproj import Transformer, CRS
    from sklearn.cluster import DBSCAN
except ImportError:
    print("AVISO: Bibliotecas geográficas (shapely, pyproj, sklearn) não encontradas. Cálculos de área limitados.")

# --- Constantes Globais ---
VELOCIDADE_LIMITE_PARADO = 0.1
DURACAO_MINIMA_PARADA_SEG = 90
MAX_PULSE_GAP_SECONDS = 180
AREA_MINIMA_BLOCO_HA = 4.0

class AnalisadorTelemetriaClima:
    """
    Versão adaptada para GitHub Actions:
    - Sem leitura de Excel.
    - Dados de estação hardcoded.
    - Execução automática (sem input).
    """
    def __init__(self, session: requests.Session):
        self.session = session
        
        # URLs da API
        self.assets_url = "https://admin.farmcommand.com/asset/?season=1083" 
        self.field_border_url = "https://admin.farmcommand.com/fieldborder/?assetID={}&format=json"
        self.canplug_url = "https://admin.farmcommand.com/canplug/?growerID={}"
        self.canplug_iot_url = "https://admin.farmcommand.com/canplug/iot/{}/?format=json"
        self.weather_url_base = "https://admin.farmcommand.com/weather/{}/historical-summary-hourly/"

        # Cache e Dados
        self.cache_limites_talhoes = {}
        self.machine_types_map = {}
        self.machine_names_map = {}
        self.fuso_horario_cuiaba = timezone('America/Cuiaba')
        self.operacoes_definidas = []
        
        # Carrega estação hardcoded
        self.estacoes_climaticas = self._carregar_estacoes_hardcoded()

    def _carregar_estacoes_hardcoded(self) -> pd.DataFrame:
        """Cria um DataFrame com os dados das estações definidos no topo do script."""
        data = ESTACOES_DO_CLIENTE
        df = pd.DataFrame(data)
        # Adiciona a coluna id_grower se não existir
        if 'id_grower' not in df.columns:
            df['id_grower'] = GROWER_ID_FIXO
        
        # Garante tipos corretos
        df['id_estacao'] = pd.to_numeric(df['id_estacao'], errors='coerce').astype('Int64')
        df['id_grower'] = pd.to_numeric(df['id_grower'], errors='coerce').astype('Int64')
        
        print(f"Dados de {len(df)} estações carregados (Hardcoded).")
        return df

    def _get_estacoes_para_produtor(self, grower_id: int) -> pd.DataFrame:
        # Retorna todas as estações (neste caso, apenas a fixa)
        # Se quiser filtrar por grower_id, pode descomentar a linha abaixo:
        # return self.estacoes_climaticas[self.estacoes_climaticas['id_grower'] == grower_id].copy()
        return self.estacoes_climaticas.copy()

    def _buscar_dados_climaticos_para_produtor(self, grower_id: int, start_date: str, end_date: str) -> pd.DataFrame:
        estacoes_produtor = self._get_estacoes_para_produtor(grower_id)
        if estacoes_produtor.empty:
            print(f"AVISO: Nenhuma estação configurada.")
            return pd.DataFrame()

        all_dfs = []
        for _, station in estacoes_produtor.iterrows():
            station_id = int(station['id_estacao'])
            station_name = station['name']
            print(f"\n--- Buscando dados climáticos para a Estação: {station_name} (ID: {station_id}) ---")
            
            raw_data = self._buscar_dados_climaticos_por_estacao(str(station_id), start_date, end_date)
            if raw_data:
                df_station = self._processar_clima_para_dataframe(raw_data, station_id, station_name)
                all_dfs.append(df_station)
        
        if not all_dfs:
            return pd.DataFrame()
            
        return pd.concat(all_dfs, ignore_index=True)

    def _buscar_dados_climaticos_por_estacao(self, station_id: str, start_date: str, end_date: str) -> list:
        all_results = []
        current_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        final_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        while current_date <= final_date:
            api_start = current_date.strftime('%Y-%m-%dT00:00:00')
            api_end = current_date.strftime('%Y-%m-%dT23:59:59')
            
            url = self.weather_url_base.format(station_id)
            params = {'startDate': api_start, 'endDate': api_end, 'format': 'json'}
            json_data = self._fazer_requisicao(url, params=params)
            
            if json_data and 'results' in json_data:
                all_results.extend(json_data['results'])
            
            time.sleep(0.1)
            current_date += timedelta(days=1)
            
        return all_results

    def _processar_clima_para_dataframe(self, json_list: list, station_id: int, station_name: str) -> pd.DataFrame:
        if not json_list: return pd.DataFrame()

        records = [{
            'datetime_utc': r.get('local_time'),
            'station_id': station_id,
            'nome_estacao': station_name,
            'temp_c': r.get('avg_temp_c'),
            'umidade_relativa': r.get('avg_relative_humidity'),
            'vento_kph': r.get('avg_windspeed_kph'),
            'delta_t': r.get('avgDeltaT'),
            'rajada_vento_kph': r.get('avgWindGust')
        } for r in json_list]
        
        df = pd.DataFrame(records)
        df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], errors='coerce', utc=True)
        df = df.dropna(subset=['datetime_utc'])
        
        for col in ['temp_c', 'umidade_relativa', 'vento_kph', 'delta_t', 'rajada_vento_kph']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['datetime_local'] = df['datetime_utc'].dt.tz_convert(self.fuso_horario_cuiaba)
        df['merge_date'] = df['datetime_local'].dt.date
        df['merge_hour'] = df['datetime_local'].dt.hour
        
        return df

    def _fazer_requisicao(self, url: str, params: dict = None) -> dict | list | None:
        for tentativa in range(2):
            try:
                response = self.session.get(url, params=params, timeout=180)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code in [401, 403]:
                    print("Sessão expirada. Tentando re-autenticar...")
                    self.session = get_authenticated_session()
                    if not self.session: return None
                    continue
                if tentativa == 0:
                    time.sleep(5)
            except json.JSONDecodeError:
                return None
        return None

    def buscar_dados_telemetria(self, implement_id: int, data_inicio: str, data_fim: str) -> dict | None:
        url = "https://admin.farmcommand.com/canplug/historical-summary/"
        params = {'endDate': data_fim, 'format': 'json', 'implementID': implement_id, 'startDate': data_inicio}
        return self._fazer_requisicao(url, params)

    def _buscar_dados_telemetria_com_chunking(self, implement_id: int, start_date_str: str, end_date_str: str) -> pd.DataFrame:
        all_dfs = []
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
        current_date = start_date
        
        print(f"  Iniciando busca detalhada (hora em hora) para evitar perda de dados...")

        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            print(f"\n  >>> Processando Dia: {date_str} (ID: {implement_id})")
            
            horas = range(24)
            for h in horas:
                start_time = f"{h:02d}:00:00"
                end_time = f"{h:02d}:59:59"
                api_start = f"{date_str}T{start_time}"
                api_end = f"{date_str}T{end_time}"
                
                sucesso = False
                tentativas = 0
                max_retries = 3
                
                while not sucesso and tentativas < max_retries:
                    try:
                        if tentativas > 0: time.sleep(3)
                        json_data = self.buscar_dados_telemetria(implement_id, api_start, api_end)
                        if json_data:
                            df_chunk = self._analisar_json_telemetria_para_df(json_data, implement_id)
                            if not df_chunk.empty:
                                all_dfs.append(df_chunk)
                                sys.stdout.write(".") 
                            else:
                                sys.stdout.write("_")
                        else:
                            sys.stdout.write("x")
                        sys.stdout.flush()
                        sucesso = True
                    except Exception as e:
                        tentativas += 1
                        time.sleep(2)
                
                if not sucesso:
                    print(f"\n      ❌ FALHA DEFINITIVA no bloco {start_time}")

            current_date += timedelta(days=1)
            
        if not all_dfs: return pd.DataFrame()
        
        print("\n  Consolidando fragmentos de dados...")
        df_combined = pd.concat(all_dfs, ignore_index=True)
        if 'Timestamp (sec)' in df_combined.columns:
            df_combined = df_combined.drop_duplicates(subset=['Timestamp (sec)', 'ImplementID'])
        df_combined['Date'] = pd.to_datetime(df_combined['Date'])
        return df_combined.sort_values(by='Timestamp (sec)').reset_index(drop=True)

    def obter_canplugs_por_produtor(self, grower_id: int) -> list:
        # Mantido para popular os mapas de nomes, mas não usado para seleção interativa
        dados_canplug = self._fazer_requisicao(self.canplug_url.format(grower_id))
        if not dados_canplug: return []
        for canplug in dados_canplug:
            implements_raw = canplug.get('implements', [])
            implements = json.loads(implements_raw) if isinstance(implements_raw, str) else implements_raw
            if not isinstance(implements, list): implements = [implements]
            implements = [int(imp) for imp in implements if str(imp).isdigit()]
            if not implements: continue
            canplug_id = canplug.get('canplugID')
            machine_type, machine_name = 'Desconhecido', 'Desconhecido'
            if canplug_id:
                iot_data = self._fazer_requisicao(self.canplug_iot_url.format(canplug_id))
                if iot_data and isinstance(iot_data.get('installed_in'), dict):
                    installed_in = iot_data['installed_in']
                    machine_name = installed_in.get('name', 'Desconhecido')
                    if isinstance(installed_in.get('machine_type'), dict):
                        machine_type = installed_in['machine_type'].get('label', 'Desconhecido')
            for imp_id in implements:
                self.machine_types_map[imp_id], self.machine_names_map[imp_id] = machine_type, machine_name
        return list(self.machine_names_map.keys())

    def obter_talhoes_por_produtor(self, grower_id: int):
        data = self._fazer_requisicao(self.assets_url)
        if not data: return
        farms = [item["id"] for item in data if item.get("category") == "Farm" and item.get("parent") == grower_id]
        fields = [item for item in data if item.get("category") == "Field" and item.get("parent") in farms]
        for talhao in fields:
            if talhao['id'] not in self.cache_limites_talhoes:
                limite = self.obter_limite_talhao(talhao['id'])
                if limite:
                    limite.update({'field_id': talhao['id'], 'field_name': talhao['label']})
                    self.cache_limites_talhoes[talhao['id']] = limite
                    
    def obter_limite_talhao(self, field_id: int) -> dict | None:
        border_data = self._fazer_requisicao(self.field_border_url.format(field_id))
        if border_data and "shapeData" in border_data[0]:
            try:
                shape_data = json.loads(border_data[0]["shapeData"])
                coords = []
                if shape_data.get("type") == "FeatureCollection":
                    geom = shape_data["features"][0].get("geometry", {})
                    if geom.get("type") == "Polygon":
                        coords = [[c[1], c[0]] for c in geom["coordinates"][0]]
                if coords:
                    return {"coordinates": coords, "field_name": border_data[0].get("label")}
            except Exception: pass
        return None

    @staticmethod
    def _is_ponto_no_poligono(ponto_lat_lon: tuple, coordenadas_poligono: list) -> bool:
        p_lat, p_lon = ponto_lat_lon
        n, dentro = len(coordenadas_poligono), False
        if n < 3: return False
        p1_lat, p1_lon = coordenadas_poligono[0]
        for i in range(n + 1):
            p2_lat, p2_lon = coordenadas_poligono[i % n]
            if p_lat > min(p1_lat, p2_lat) and p_lat <= max(p1_lat, p2_lat) and p_lon <= max(p1_lon, p2_lon) and p1_lat != p2_lat:
                xinters = (p_lat - p1_lat) * (p2_lon - p1_lon) / (p2_lat - p1_lat) + p1_lon
                if p1_lon == p2_lon or p_lon <= xinters: dentro = not dentro
            p1_lat, p1_lon = p2_lat, p2_lon
        return dentro

    def _get_first_valid_value(self, data: dict, keys: list) -> any:
        for key in keys:
            value = data.get(key)
            if value is not None: return value
        return None
    
    def _analisar_json_telemetria_para_df(self, json_data: any, implement_id: int) -> pd.DataFrame:
        features_list = []
        if isinstance(json_data, dict) and 'results' in json_data:
            for resultado in json_data.get('results', []):
                if resultado.get('type') == 'FeatureCollection':
                    features_list.extend(resultado.get('features', []))
        elif isinstance(json_data, list):
            features_list = json_data
        
        if not features_list: return pd.DataFrame()

        registros = []
        for feature in features_list:
            if isinstance(feature, dict):
                coords = None
                if 'geometry' in feature and feature['geometry']:
                    coords = feature['geometry'].get('coordinates')
                
                if not coords or len(coords) < 2: continue
                
                props = feature.get('properties', {}).copy()
                props['Longitude'], props['Latitude'] = coords[0], coords[1]
                
                for key in list(props.keys()):
                    if isinstance(props[key], dict):
                        props[key] = props[key].get('value', props[key].get('status'))

                vel_raw = self._get_first_valid_value(props, ['Computed Velocity (miles/hour)', 'velocity', 'Ground Speed (miles/hour)'])
                vel_kmh = pd.to_numeric(vel_raw, errors='coerce')
                if vel_raw is not None: vel_kmh = float(vel_raw) * 1.60934
                props['velocity'] = vel_kmh if pd.notna(vel_kmh) and vel_kmh <= 60 else np.nan

                props['Fuel Rate (L/h)'] = self._get_first_valid_value(props, ['Fuel Rate (L/h)', 'Fuel Consumption (L/h)', 'Instantaneous Liquid Fuel Usage (L/hour)'])
                width_m = self._get_first_valid_value(props, ['machine width (meters)', 'Implement Width (meters)'])
                if width_m is None and 'Header width in use (mm)' in props:
                    width_m = props.get('Header width in use (mm)', 0) / 1000
                props['machine width (meters)'] = width_m
                
                registros.append(props)

        if not registros: return pd.DataFrame()
        df = pd.DataFrame(registros)

        df['Datetime'] = pd.to_datetime(df['Timestamp (sec)'], unit='s', utc=True).dt.tz_convert(self.fuso_horario_cuiaba)
        df['Date'] = df['Datetime'].dt.date
        df['ImplementID'] = implement_id
        df['MachineName'] = self.machine_names_map.get(implement_id, f"ID {implement_id}")
        
        cols_num = ['velocity', 'Fuel Rate (L/h)', 'Engine RPM', 'machine width (meters)']
        for col in cols_num:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['Inside_Field_Name'] = pd.NA
        if self.cache_limites_talhoes:
            df_valid = df.dropna(subset=['Latitude', 'Longitude'])
            for idx, row in df_valid.iterrows():
                for border in self.cache_limites_talhoes.values():
                    if border.get('coordinates') and self._is_ponto_no_poligono((row['Latitude'], row['Longitude']), border['coordinates']):
                        df.loc[idx, 'Inside_Field_Name'] = border['field_name']
                        break
        return df

    @staticmethod
    def _haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371
        lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
        lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def _encontrar_estacao_mais_proxima(self, lat, lon, estacoes_df):
        if estacoes_df.empty or pd.isna(lat) or pd.isna(lon): return None
        distancias = estacoes_df.apply(lambda row: self._haversine_distance(lat, lon, row['latitude'], row['longitude']), axis=1)
        return estacoes_df.loc[distancias.idxmin()]['id_estacao']
        
    def _merge_telemetria_clima(self, df_telemetria: pd.DataFrame, df_clima: pd.DataFrame, estacoes_produtor: pd.DataFrame) -> pd.DataFrame:
        print("\nCruzando dados de telemetria e clima...")
        if df_clima.empty or df_telemetria.empty:
            df_telemetria['temp_c'] = np.nan
            df_telemetria['umidade_relativa'] = np.nan
            df_telemetria['vento_kph'] = np.nan
            df_telemetria['delta_t'] = np.nan
            df_telemetria['rajada_vento_kph'] = np.nan
            df_telemetria['CondicaoAplicacao'] = 'Sem Dados'
            return df_telemetria
        
        df_telemetria['nearest_station_id'] = df_telemetria.apply(
            lambda row: self._encontrar_estacao_mais_proxima(row['Latitude'], row['Longitude'], estacoes_produtor),
            axis=1
        ).astype('float64')

        df_telemetria['merge_date'] = df_telemetria['Datetime'].dt.date
        df_telemetria['merge_hour'] = df_telemetria['Datetime'].dt.hour
        
        df_merged = pd.merge(
            df_telemetria,
            df_clima[['station_id', 'merge_date', 'merge_hour', 'temp_c', 'umidade_relativa', 'vento_kph', 'delta_t', 'rajada_vento_kph']],
            how='left',
            left_on=['nearest_station_id', 'merge_date', 'merge_hour'],
            right_on=['station_id', 'merge_date', 'merge_hour']
        )
        
        def get_delta_t_state(delta_t_val):
            if pd.isna(delta_t_val): return 'Sem Dados'
            if delta_t_val >= 9: return 'Vermelho'
            if delta_t_val < 2 or (delta_t_val > 8 and delta_t_val < 9): return 'Amarelo'
            if delta_t_val >= 2 and delta_t_val <= 8: return 'Verde'
            return 'Sem Dados'

        def get_vento_state(vento_val):
            if pd.isna(vento_val): return 'Sem Dados'
            if (vento_val >= 0 and vento_val <= 1) or (vento_val > 10): return 'Vermelho'
            if (vento_val > 1 and vento_val <= 2) or (vento_val > 8 and vento_val <= 10): return 'Amarelo'
            if vento_val > 2 and vento_val <= 8: return 'Verde'
            return 'Sem Dados'

        def get_condicao_aplicacao(row):
            vento_state = get_vento_state(row['vento_kph']) 
            delta_t_state = get_delta_t_state(row['delta_t'])
            if vento_state == 'Vermelho' or delta_t_state == 'Vermelho': return 'Evitar'
            if vento_state == 'Verde' and delta_t_state == 'Verde': return 'Ideal'
            if vento_state == 'Amarelo' or delta_t_state == 'Amarelo': return 'Atenção'
            return 'Sem Dados'
            
        df_merged['CondicaoAplicacao'] = df_merged.apply(get_condicao_aplicacao, axis=1)
        return df_merged.drop(columns=['merge_date', 'merge_hour', 'nearest_station_id', 'station_id'])

    def _get_utm_projection(self, df: pd.DataFrame) -> CRS | None:
        if df.empty or df['Longitude'].isnull().all() or df['Latitude'].isnull().all(): return None
        avg_lon = df['Longitude'].mean()
        avg_lat = df['Latitude'].mean()
        utm_zone = math.floor((avg_lon + 180) / 6) + 1
        return CRS(f"EPSG:327{utm_zone}") if avg_lat < 0 else CRS(f"EPSG:326{utm_zone}")

    def _estimar_largura_por_geometria(self, df_maquina: pd.DataFrame) -> float | None:
        df_trabalho = df_maquina[df_maquina['Operating_Mode'] == 'Trabalho Produtivo'].copy()
        if len(df_trabalho) < 50: return None
        try:
            crs_wgs84 = CRS("EPSG:4326")
            crs_utm = self._get_utm_projection(df_trabalho)
            if not crs_utm: return None
            transformer = Transformer.from_crs(crs_wgs84, crs_utm, always_xy=True)
            points_utm = [transformer.transform(lon, lat) for lon, lat in df_trabalho[['Longitude', 'Latitude']].values]
            df_trabalho[['utm_x', 'utm_y']] = points_utm
        except Exception: return None

        coords = df_trabalho[['utm_x', 'utm_y']].values
        df_trabalho['pass_label'] = DBSCAN(eps=10, min_samples=5).fit(coords).labels_
        pass_labels = [l for l in df_trabalho['pass_label'].unique() if l != -1]
        if len(pass_labels) < 2: return None

        pass_lines = {l: LineString(df_trabalho[df_trabalho['pass_label'] == l][['utm_x', 'utm_y']].values) for l in pass_labels if len(df_trabalho[df_trabalho['pass_label'] == l]) > 1}
        measured_widths = []
        labels_list = list(pass_lines.keys())
        for i in range(len(labels_list)):
            for j in range(i + 1, len(labels_list)):
                line1, line2 = pass_lines[labels_list[i]], pass_lines[labels_list[j]]
                for k in range(5):
                    dist = line1.interpolate(k/4, normalized=True).distance(line2)
                    if 5 < dist < 50: measured_widths.append(dist)

        if not measured_widths: return None
        hist, bin_edges = np.histogram(measured_widths, bins=np.arange(min(measured_widths), max(measured_widths) + 1, 0.5))
        if hist.any(): return bin_edges[np.argmax(hist)]
        return None

    def _identificar_blocos_operacao(self, df: pd.DataFrame):
        df_trabalho = df[df['Operating_Mode'] == 'Trabalho Produtivo'].copy()
        if df_trabalho.empty: return

        df_trabalho['DateOnly'] = pd.to_datetime(df_trabalho['Date']).dt.date
        df_trabalho = df_trabalho.sort_values(by=['ImplementID', 'Inside_Field_Name', 'DateOnly'])
        df_trabalho['date_diff'] = df_trabalho.groupby(['ImplementID', 'Inside_Field_Name'])['DateOnly'].diff().apply(lambda x: x.days if pd.notna(x) else 0)
        df_trabalho['operation_block'] = (df_trabalho['date_diff'] > 1).cumsum()

        blocos = df_trabalho.groupby(['ImplementID', 'Inside_Field_Name', 'operation_block'])
        op_counter = 1
        for (imp_id, talhao, _), grupo in blocos:
            if pd.isna(talhao): continue
            start_date_str = grupo['DateOnly'].min().strftime('%d/%m')
            maquina = self.machine_names_map.get(imp_id, f"ID {imp_id}")
            nome_bloco = f"Bloco {op_counter}: {maquina.split()[0]} em {talhao} (a partir de {start_date_str})"
            
            self.operacoes_definidas.append({'id': op_counter, 'nome': nome_bloco, 'tipo': "Não definido", 'indices_df': grupo.index.tolist()})
            df.loc[grupo.index, 'OperationID'] = op_counter
            df.loc[grupo.index, 'OperationName'] = nome_bloco
            op_counter += 1

    def executar_relatorio(self):
        # --- CONFIGURAÇÃO AUTOMÁTICA DE DATAS ---
        # Hoje
        hoje = datetime.now()
        # 30 dias atrás
        inicio = hoje - timedelta(days=5)
        
        data_inicio = inicio.strftime('%Y-%m-%d')
        data_fim = hoje.strftime('%Y-%m-%d')
        
        grower_id = GROWER_ID_FIXO
        implement_ids = IMPLEMENT_IDS_FIXO
        
        print(f"\n--- INICIANDO EXECUÇÃO AUTOMÁTICA ---")
        print(f"Período: {data_inicio} até {data_fim}")
        print(f"Produtor ID: {grower_id}")
        
        self.obter_talhoes_por_produtor(grower_id)
        # Popula nomes e tipos das máquinas
        self.obter_canplugs_por_produtor(grower_id)

        # Lógica de Seleção de Máquinas
        if IMPLEMENT_IDS_FIXO:
            implement_ids = IMPLEMENT_IDS_FIXO
            print(f"Usando lista fixa de máquinas: {implement_ids}")
        else:
            print("Lista fixa vazia. Buscando máquinas do tipo 'Sprayer'/'Pulverizador'...")
            implement_ids = []
            for mid, mtype in self.machine_types_map.items():
                # Verifica se algum dos tipos alvo está contido no tipo da máquina (case insensitive)
                if any(alvo.lower() in mtype.lower() for alvo in TIPO_MAQUINA_ALVO):
                    implement_ids.append(mid)
                    print(f"  -> Encontrada: ID {mid} ({self.machine_names_map.get(mid)}) - Tipo: {mtype}")
            
            if not implement_ids:
                print("AVISO: Nenhuma máquina do tipo 'Sprayer' encontrada. Tentando processar tudo ou encerrando...")
                # Opcional: Se não achar nada, pode encerrar ou pegar tudo. 
                # Vou encerrar para evitar processar tratores/colheitadeiras sem querer.
                print("Nenhuma máquina alvo encontrada. Encerrando.")
                return None
        
        print(f"Máquinas selecionadas para processamento: {implement_ids}")
        
        df_clima_completo = self._buscar_dados_climaticos_para_produtor(grower_id, data_inicio, data_fim)
        estacoes_deste_produtor = self._get_estacoes_para_produtor(grower_id)

        dfs_telemetria = [df for df in [self._buscar_dados_telemetria_com_chunking(imp, data_inicio, data_fim) for imp in implement_ids] if not df.empty]
        if not dfs_telemetria:
            print("Nenhum dado de telemetria carregado. Finalizando.")
            return None
        df_telemetria_completo = pd.concat(dfs_telemetria, ignore_index=True).sort_values('Timestamp (sec)').reset_index(drop=True)
        
        df_final_combinado = self._merge_telemetria_clima(df_telemetria_completo, df_clima_completo, estacoes_deste_produtor)

        df_final_combinado['Operating_Mode'] = np.select(
            [df_final_combinado['velocity'] <= VELOCIDADE_LIMITE_PARADO, df_final_combinado['Inside_Field_Name'].notna()],
            ['Parado', 'Trabalho Produtivo'], default='Deslocamento'
        )

        for imp_id in df_final_combinado['ImplementID'].unique():
            mask = df_final_combinado['ImplementID'] == imp_id
            larguras_api = df_final_combinado.loc[mask, 'machine width (meters)'].dropna()
            larguras_api = larguras_api[larguras_api > 0.1]
            largura_final = np.nan
            if not larguras_api.empty:
                largura_final = Counter(larguras_api).most_common(1)[0][0]
            else:
                largura_est = self._estimar_largura_por_geometria(df_final_combinado[mask])
                if largura_est: largura_final = largura_est
            df_final_combinado.loc[mask, 'machine width (meters)'] = largura_final

        df_final_combinado['OperationID'] = pd.NA
        df_final_combinado['OperationName'] = pd.NA
        self._identificar_blocos_operacao(df_final_combinado)

        print("\nProcessamento concluído. Gerando relatório HTML...")
        self.criar_relatorio_html_unificado(df_final_combinado, data_inicio, data_fim)
        
        return df_final_combinado

    def criar_relatorio_html_unificado(self, df: pd.DataFrame, data_inicio_str: str, data_fim_str: str):
        if df.empty: return
        # (Aqui iria o código de geração de HTML, mantendo simples para o exemplo)
        # Vou salvar um HTML básico para provar que funcionou
        filename = "relatorio_telemetria_clima.html"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"<h1>Relatório Gerado em {datetime.now()}</h1>")
            f.write(f"<p>Período: {data_inicio_str} a {data_fim_str}</p>")
            f.write(f"<p>Total de registros: {len(df)}</p>")
            f.write(df.head(100).to_html())
        print(f"Relatório salvo em: {os.path.abspath(filename)}")

# --- PONTO DE ENTRADA PRINCIPAL DO SCRIPT ---
if __name__ == "__main__":
    print("Iniciando autenticação via farm_auth...")
    sessao_autenticada = get_authenticated_session()
    
    if not sessao_autenticada:
        print("❌ ERRO CRÍTICO: Falha na autenticação. Encerrando.")
        sys.exit(1)
    print("Autenticação principal bem-sucedida.")
    
    analisador = AnalisadorTelemetriaClima(session=sessao_autenticada)
    analisador.executar_relatorio()
